''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import re
import os
import progressbar
import warnings
from collections import Iterable

import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import tables as tb
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, leastsq, basinhopping, OptimizeWarning, minimize
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import data_selection

# Imports for track based alignment
from testbeam_analysis.track_analysis import fit_tracks
from testbeam_analysis.result_analysis import calculate_residuals

warnings.simplefilter("ignore", OptimizeWarning)  # Fit errors are handled internally, turn of warnings


def correlate_cluster(input_cluster_files, output_correlation_file, n_pixels, pixel_size=None, dut_names=None, chunk_size=4999999):
    '''Histograms the cluster column (row) of two different devices on an event basis.

    If the cluster positions are correlated a line should be seen. The cluster positions are round to 1 um precision to increase the histogramming speed.
    All permutations are considered (all cluster of the first device are correlated with all cluster of the second device).

    Parameters
    ----------
    input_cluster_files : iterable of pytables file
        Input files with cluster data. One file per DUT.
    output_correlation_file : pytables file
        Output file with the correlation histograms.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension in um in column, row direction
        e.g. for 2 DUTs: pixel_size = [(250, 50), (250, 50)]
    dut_names : iterable of strings
        To show the DUT names in the plot
    chunk_size: int
        Defines the amount of in-RAM data. The higher the more RAM is used and the faster this function works.
    '''

    logging.info('=== Correlate the position of %d DUTs ===', len(input_cluster_files))

    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        n_duts = len(input_cluster_files)

        # Result arrays to be filled
        column_correlations = []
        row_correlations = []
        for dut_index in range(1, n_duts):
            shape_column = (n_pixels[dut_index][0], n_pixels[0][0])
            shape_row = (n_pixels[dut_index][1], n_pixels[0][1])
            column_correlations.append(np.zeros(shape_column, dtype=np.int))
            row_correlations.append(np.zeros(shape_row, dtype=np.int))

        start_indices = [0] * (n_duts - 1)  # Store the loop indices for speed up

        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Cluster.shape[0], term_width=80)
            progress_bar.start()

            pool = Pool()  # Provide worker pool
            for cluster_dut_0, index in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = cluster_dut_0[:]['event_number']

                # Create correlation histograms to the reference device for all other devices
                # Do this in parallel to safe time

                dut_results = []
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    dut_results.append(pool.apply_async(_correlate_cluster, kwds={'cluster_dut_0': cluster_dut_0,
                                                                                  'cluster_file': cluster_file,
                                                                                  'start_index': start_indices[dut_index - 1],
                                                                                  'start_event_number': actual_event_numbers[0],
                                                                                  'stop_event_number': actual_event_numbers[-1] + 1,
                                                                                  'column_correlation': column_correlations[dut_index - 1],
                                                                                  'row_correlation': row_correlations[dut_index - 1],
                                                                                  'chunk_size': chunk_size
                                                                                  }
                                                        ))
                # Collect results when available
                for dut_index, dut_result in enumerate(dut_results, start=1):
                    (start_indices[dut_index - 1], column_correlations[dut_index - 1], row_correlations[dut_index - 1]) = dut_result.get()

                progress_bar.update(index)

            pool.close()
            pool.join()

        # Store the correlation histograms
        for dut_index in range(n_duts - 1):
            out_col = out_file_h5.create_carray(out_file_h5.root, name='CorrelationColumn_%d_0' % (dut_index + 1), title='Column Correlation between DUT %d and %d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(column_correlations[dut_index].dtype), shape=column_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            out_row = out_file_h5.create_carray(out_file_h5.root, name='CorrelationRow_%d_0' % (dut_index + 1), title='Row Correlation between DUT %d and %d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(row_correlations[dut_index].dtype), shape=row_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            out_col.attrs.filenames = [str(input_cluster_files[0]), str(input_cluster_files[dut_index])]
            out_row.attrs.filenames = [str(input_cluster_files[0]), str(input_cluster_files[dut_index])]
            out_col[:] = column_correlations[dut_index]
            out_row[:] = row_correlations[dut_index]
        progress_bar.finish()

    plot_utils.plot_correlations(input_correlation_file=output_correlation_file, pixel_size=pixel_size, dut_names=dut_names)


def merge_cluster_data(input_cluster_files, output_merged_file, n_pixels, pixel_size, chunk_size=4999999):
    '''Takes the cluster from all cluster files and merges them into one big table aligned at a common event number.
    Empty entries are signaled with column = row = charge = nan. Position is translated from indices to um. The
    local coordinate system rigin (0, 0) is defined in the sensor center, to decouple translation and rotation.

    Alignment information from the alignment file is used to correct the column/row positions. Use alignment data if
    available (translation/rotation for each plane), otherwise pre-alignment data (offset, slope of correlation)
    will be used.

    Parameters
    ----------
    input_cluster_files : list of pytables files
        File name of the input cluster files with correlation data.
    output_merged_file : pytables file
        File name of the output tracklet file.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension in um in column, row direction
        e.g. for 2 DUTs: pixel_size = [(250, 50), (250, 50)]
    chunk_size: int
        Defines the amount of in RAM data. The higher the more RAM is used and the faster this function works.
    '''
    logging.info('=== Merge cluster from %d DUTSs to merged hit file ===', len(input_cluster_files))

    # Create result array description, depends on the number of DUTs
    description = [('event_number', np.int64)]
    for index, _ in enumerate(input_cluster_files):
        description.append(('x_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('y_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('z_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('charge_dut_%d' % index, np.float))
    description.extend([('track_quality', np.uint32), ('n_tracks', np.int8)])

    start_indices = [0] * len(input_cluster_files)  # Store the loop indices for speed up
    start_indices_2 = [0] * len(input_cluster_files)  # Additional indices for second loop

    # Merge the cluster data from different DUTs into one table
    with tb.open_file(output_merged_file, mode='w') as out_file_h5:
        merged_cluster_table = out_file_h5.create_table(out_file_h5.root, name='MergedCluster', description=np.zeros((1,), dtype=description).dtype, title='Merged cluster on event number', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Cluster.shape[0], term_width=80)
            progress_bar.start()
            actual_start_event_number = 0  # Defines the first event number of the actual chunk for speed up. Cannot be deduced from DUT0, since this DUT could have missing event numbers.
            for cluster_dut_0, index in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = cluster_dut_0[:]['event_number']

                # First loop: calculate the minimum event number indices needed to merge all cluster from all files to this event number index
                common_event_numbers = actual_event_numbers
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open DUT0 cluster file
                        for actual_cluster, start_indices[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start=start_indices[dut_index], start_event_number=actual_start_event_number, stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size):  # Loop over the cluster in the actual cluster file in chunks
                            common_event_numbers = analysis_utils.get_max_events_in_both_arrays(common_event_numbers, actual_cluster[:]['event_number'])
                merged_cluster_array = np.empty((common_event_numbers.shape[0],), dtype=description)  # Result array to be filled. For no hit: column = row = NaN
                merged_cluster_array[:] = np.nan

                # Set the event number
                merged_cluster_array['event_number'] = common_event_numbers[:]

                # Fill result array with DUT 0 data
                actual_cluster = analysis_utils.map_cluster(common_event_numbers, cluster_dut_0)
                # Add only real hits, nan is a virtual hit
                selection = ~np.isnan(actual_cluster['mean_column'])
                # Convert indices to positions, origin defined in the center of the sensor
                merged_cluster_array['x_dut_0'][selection] = pixel_size[0][0] * (actual_cluster['mean_column'][selection] - 0.5 - (0.5 * n_pixels[0][0]))
                merged_cluster_array['y_dut_0'][selection] = pixel_size[0][1] * (actual_cluster['mean_row'][selection] - 0.5 - (0.5 * n_pixels[0][1]))
                merged_cluster_array['z_dut_0'][selection] = 0.0
                merged_cluster_array['charge_dut_0'][selection] = actual_cluster['charge'][selection]

                # Fill result array with other DUT data
                # Second loop: get the cluster from all files and merge them to the common event number
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_cluster, start_indices_2[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start=start_indices_2[dut_index], start_event_number=common_event_numbers[0], stop_event_number=common_event_numbers[-1] + 1, chunk_size=chunk_size):  # Loop over the cluster in the actual cluster file in chunks
                            actual_cluster = analysis_utils.map_cluster(common_event_numbers, actual_cluster)
                            # Add only real hits, nan is a virtual hit
                            selection = ~np.isnan(actual_cluster['mean_column'])
                            # Convert indices to positions, origin in the center of the sensor
                            actual_mean_column = pixel_size[dut_index][0] * (actual_cluster['mean_column'][selection] - 0.5 - (0.5 * n_pixels[dut_index][0]))
                            actual_mean_row = pixel_size[dut_index][1] * (actual_cluster['mean_row'][selection] - 0.5 - (0.5 * n_pixels[dut_index][1]))

                            merged_cluster_array['x_dut_%d' % (dut_index)][selection] = actual_mean_column
                            merged_cluster_array['y_dut_%d' % (dut_index)][selection] = actual_mean_row
                            merged_cluster_array['z_dut_%d' % (dut_index)][selection] = 0.0

                            merged_cluster_array['charge_dut_%d' % (dut_index)][selection] = actual_cluster['charge'][selection]

                merged_cluster_table.append(merged_cluster_array)
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(index)
            progress_bar.finish()


def prealignment(input_correlation_file, output_alignment_file, z_positions, pixel_size, s_n=0.1, fit_background=False, reduce_background=False, dut_names=None, no_fit=False, non_interactive=True, iterations=2):
    '''Deduce a pre-alignment from the correlations, by fitting the correlations with a straight line (gives offset, slope, but no tild angles).
       The user can define cuts on the fit error and straight line offset in an interactive way.

    Parameters
    ----------
    input_correlation_file : pytbales file
        The input file with the correlation histograms.
    output_alignment_file : pytables file
        The output file for correlation data.
    z_positions : iterable
        The positions of the devices in z in um
    s_n : float
        The signal to noise ratio for peak signal over background peak. This should be specified when the background is fitted with a gaussian.
        Usually data with a lot if tracks per event have a gaussian background. The S/N can be guesses by looking at the correlation plot. The std.
        value is usually fine.
    fit_background : bool
        Data with a lot if tracks per event have a gaussian background from the beam profile. Also try to fit this background to determine the correlation
        peak correctly. If you see a clear 2D gaussian in the correlation plot this shoud be activated. If you have 1-2 tracks per event and large pixels
        this option should be off, because otherwise overfitting is possible.
    reduce_background : bool
        Reduce background (uncorrelated events) with the help of SVD.
    no_fit : bool
        Use Hough transformation to calculate slope and offset.
    pixel_size: iterable
        Iterable of tuples with column and row pixel size in um
    dut_names: iterable
        List of names of the DUTs.
    non_interactive : boolean
        Deactivate user interaction and apply cuts automatically
    iterations : number
        Only used in non interactive mode. Sets how often automatic cuts are applied.
    '''
    logging.info('=== Pre-alignment ===')

    if no_fit:
        if not reduce_background:
            logging.warning("no_fit is True, setting reduce_background to True")
            reduce_background = True

    if reduce_background:
        if fit_background:
            logging.warning("reduce_background is True, setting fit_background to False")
            fit_background = False

    with PdfPages(os.path.join(os.path.dirname(os.path.abspath(output_alignment_file)), 'Prealignment.pdf')) as output_pdf:
        with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
            n_duts = len(in_file_h5.list_nodes("/")) // 2 + 1  # no correlation for reference DUT0
            result = np.zeros(shape=(n_duts,), dtype=[('DUT', np.uint8), ('column_c0', np.float), ('column_c0_error', np.float), ('column_c1', np.float), ('column_c1_error', np.float), ('column_sigma', np.float), ('column_sigma_error', np.float), ('row_c0', np.float), ('row_c0_error', np.float), ('row_c1', np.float), ('row_c1_error', np.float), ('row_sigma', np.float), ('row_sigma_error', np.float), ('z', np.float)])
            # Set std. settings for reference DUT0
            result[0]['column_c0'], result[0]['column_c0_error'] = 0.0, 0.0
            result[0]['column_c1'], result[0]['column_c1_error'] = 1.0, 0.0
            result[0]['row_c0'], result[0]['row_c0_error'] = 0.0, 0.0
            result[0]['row_c1'], result[0]['row_c1_error'] = 1.0, 0.0
            result[0]['z'] = z_positions[0]
            for node in in_file_h5.root:
                table_prefix = 'column' if 'column' in node.name.lower() else 'row'
                indices = re.findall(r'\d+', node.name)
                dut_idx = int(indices[0])
                ref_idx = int(indices[1])
                result[dut_idx]['DUT'] = dut_idx
                dut_name = dut_names[dut_idx] if dut_names else ("DUT " + str(dut_idx))
                ref_name = dut_names[ref_idx] if dut_names else ("DUT " + str(ref_idx))
                logging.info('Aligning data from %s', node.name)

                if "column" in node.name.lower():
                    pixel_size_dut, pixel_size_ref = pixel_size[dut_idx][0], pixel_size[ref_idx][0]
                else:
                    pixel_size_dut, pixel_size_ref = pixel_size[dut_idx][1], pixel_size[ref_idx][1]

                data = node[:]

                n_pixel_dut, n_pixel_ref = data.shape[0], data.shape[1]

                # Initialize arrays with np.nan (invalid), adding 0.5 to change from index to position
                # matrix index 0 is cluster index 1 ranging from 0.5 to 1.4999, which becomes position 0.0 to 0.999 with center at 0.5, etc.
                x_ref = (np.linspace(0.0, n_pixel_ref, num=n_pixel_ref, endpoint=False, dtype=np.float) + 0.5)
                x_dut = (np.linspace(0.0, n_pixel_dut, num=n_pixel_dut, endpoint=False, dtype=np.float) + 0.5)
                coeff_fitted = [None] * n_pixel_dut
                mean_fitted = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Peak of the Gauss fit
                mean_fitted.fill(np.nan)
                mean_error_fitted = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Error of the fit of the peak
                mean_error_fitted.fill(np.nan)
                sigma_fitted = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Sigma of the Gauss fit
                sigma_fitted.fill(np.nan)
                chi2 = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Chi2 of the fit
                chi2.fill(np.nan)
                n_cluster = np.sum(data, axis=1)  # Number of hits per bin

                if reduce_background:
                    uu, dd, vv = np.linalg.svd(data)  # sigular value decomposition
                    background = np.matrix(uu[:, :1]) * np.diag(dd[:1]) * np.matrix(vv[:1, :])  # take first sigular value for background
                    background = np.array(background, dtype=np.int)  # make Numpy array
                    data = (data - background).astype(np.int)  # remove background
                    data -= data.min()  # only positive values

                if no_fit:
                    # calculate half hight
                    median = np.median(data)
                    median_max = np.median(np.max(data, axis=1))
                    half_median_data = (data > ((median + median_max) / 2))
                    # calculate maximum per column
                    max_select = np.argmax(data, axis=1)
                    hough_data = np.zeros_like(data)
                    hough_data[np.arange(data.shape[0]), max_select] = 1
                    # select maximums if larger than half hight
                    hough_data = hough_data & half_median_data
                    # transpose for correct angle
                    hough_data = hough_data.T
                    accumulator, theta, rho, theta_edges, rho_edges = analysis_utils.hough_transform(hough_data, theta_res=0.1, rho_res=1.0, return_edges=True)
                    rho_idx, th_idx = np.unravel_index(accumulator.argmax(), accumulator.shape)
                    rho_val, theta_val = rho[rho_idx], theta[th_idx]
                    slope_idx, offset_idx = -np.cos(theta_val) / np.sin(theta_val), rho_val / np.sin(theta_val)
                    slope = slope_idx * (pixel_size_ref / pixel_size_dut)
                    offset = offset_idx * pixel_size_ref
                    # offset in the center of the pixel matrix
                    offset_center = offset + slope * pixel_size_dut * n_pixel_dut * 0.5 - pixel_size_ref * n_pixel_ref * 0.5
                    offset_center += 0.5 * pixel_size_ref - slope * 0.5 * pixel_size_dut  # correct for half bin

                    result[dut_idx][table_prefix + '_c0'], result[dut_idx][table_prefix + '_c0_error'] = offset_center, 0.0
                    result[dut_idx][table_prefix + '_c1'], result[dut_idx][table_prefix + '_c1_error'] = slope, 0.0
                    result[dut_idx][table_prefix + '_sigma'], result[dut_idx][table_prefix + '_sigma_error'] = 0.0, 0.0
                    result[dut_idx]['z'] = z_positions[dut_idx]

                    plot_utils.plot_hough(
                        x=x_dut,
                        data=hough_data,
                        accumulator=accumulator,
                        offset=offset_idx,
                        slope=slope_idx,
                        theta_edges=theta_edges,
                        rho_edges=rho_edges,
                        n_pixel_ref=n_pixel_ref,
                        n_pixel_dut=n_pixel_dut,
                        pixel_size_ref=pixel_size_ref,
                        pixel_size_dut=pixel_size_dut,
                        ref_name=ref_name,
                        dut_name=dut_name,
                        prefix=table_prefix,
                        output_pdf=output_pdf)

                else:
                    # fill the arrays from above with values
                    _fit_data(x=x_ref, data=data, s_n=s_n, coeff_fitted=coeff_fitted, mean_fitted=mean_fitted, mean_error_fitted=mean_error_fitted, sigma_fitted=sigma_fitted, chi2=chi2, fit_background=fit_background, reduce_background=reduce_background)

                    # Convert fit results to metric units for alignment fit
                    # Origin is center of pixel matrix
                    x_dut_scaled = (x_dut - 0.5 * n_pixel_dut) * pixel_size_dut
                    mean_fitted_scaled = (mean_fitted - 0.5 * n_pixel_ref) * pixel_size_ref
                    mean_error_fitted_scaled = mean_error_fitted * pixel_size_ref

                    # Selected data arrays
                    x_selected = x_dut.copy()
                    x_dut_scaled_selected = x_dut_scaled.copy()
                    mean_fitted_scaled_selected = mean_fitted_scaled.copy()
                    mean_error_fitted_scaled_selected = mean_error_fitted_scaled.copy()
                    sigma_fitted_selected = sigma_fitted.copy()
                    chi2_selected = chi2.copy()
                    n_cluster_selected = n_cluster.copy()

                    # Show the straigt line correlation fit including fit errors and offsets from the fit
                    # Let the user change the cuts (error limit, offset limit) and refit until result looks good
                    refit = True
                    selected_data = np.ones_like(x_dut, dtype=np.bool)
                    actual_iteration = 0  # Refit counter for non interactive mode
                    while(refit):
                        selected_data, fit, refit = plot_utils.plot_alignments(x=x_dut_scaled_selected,
                                                                               mean_fitted=mean_fitted_scaled_selected,
                                                                               mean_error_fitted=mean_error_fitted_scaled_selected,
                                                                               n_cluster=n_cluster_selected,
                                                                               ref_name=ref_name,
                                                                               dut_name=dut_name,
                                                                               prefix=table_prefix,
                                                                               non_interactive=non_interactive)
                        x_selected = x_selected[selected_data]
                        x_dut_scaled_selected = x_dut_scaled_selected[selected_data]
                        mean_fitted_scaled_selected = mean_fitted_scaled_selected[selected_data]
                        mean_error_fitted_scaled_selected = mean_error_fitted_scaled_selected[selected_data]
                        sigma_fitted_selected = sigma_fitted_selected[selected_data]
                        chi2_selected = chi2_selected[selected_data]
                        n_cluster_selected = n_cluster_selected[selected_data]
                        # Stop in non interactive mode if the number of refits (iterations) is reached
                        if non_interactive:
                            actual_iteration += 1
                            if actual_iteration > iterations:
                                break

                    # Linear fit, usually describes correlation very well, slope is close to 1.
                    # With low energy beam and / or beam with diverse agular distribution, the correlation will not be perfectly straight
                    # Use results from straight line fit as start values for this final fit
                    re_fit, re_fit_pcov = curve_fit(analysis_utils.linear, x_dut_scaled_selected, mean_fitted_scaled_selected, sigma=mean_error_fitted_scaled_selected, absolute_sigma=True, p0=[fit[0], fit[1]])

                    # Write fit results to array
                    result[dut_idx][table_prefix + '_c0'], result[dut_idx][table_prefix + '_c0_error'] = re_fit[0], np.absolute(re_fit_pcov[0][0]) ** 0.5
                    result[dut_idx][table_prefix + '_c1'], result[dut_idx][table_prefix + '_c1_error'] = re_fit[1], np.absolute(re_fit_pcov[1][1]) ** 0.5
                    result[dut_idx]['z'] = z_positions[dut_idx]

                    # Calculate mean sigma (is a residual when assuming straight tracks) and its error and store the actual data in result array
                    # This error is needed for track finding and track quality determination
                    mean_sigma = pixel_size_ref * np.mean(np.array(sigma_fitted_selected))
                    mean_sigma_error = pixel_size_ref * np.std(np.array(sigma_fitted_selected)) / np.sqrt(np.array(sigma_fitted_selected).shape[0])

                    result[dut_idx][table_prefix + '_sigma'], result[dut_idx][table_prefix + '_sigma_error'] = mean_sigma, mean_sigma_error

                    # Calculate the index of the beam center based on valid indices
                    plot_index = np.average(x_selected - 1, weights=np.sum(data, axis=1)[np.array(x_selected - 1, dtype=np.int)])
                    # Find nearest valid index to the calculated index
                    idx = (np.abs(x_selected - 1 - plot_index)).argmin()
                    plot_index = np.array(x_selected - 1, dtype=np.int)[idx]

                    if np.all(np.isnan(coeff_fitted[plot_index][3:6])):
                        y_fit = analysis_utils.gauss_offset(x_ref, *coeff_fitted[plot_index][[0, 1, 2, 6]])
                        fit_label = "Gauss-Offset"
                    else:
                        y_fit = analysis_utils.double_gauss_offset(x_ref, *coeff_fitted[plot_index])
                        fit_label = "Gauss-Gauss-Offset"
                    plot_utils.plot_correlation_fit(x=x_ref,
                                                    y=data[plot_index, :],
                                                    y_fit=y_fit,
                                                    xlabel='%s %s' % ("Column" if "column" in node.name.lower() else "Row", ref_name),
                                                    fit_label=fit_label,
                                                    title="Correlation of %s: %s vs. %s at %s %d" % (table_prefix + "s", ref_name, dut_name, table_prefix, plot_index),
                                                    output_pdf=output_pdf)

                    # Plot selected data with fit
                    fit_fn = np.poly1d(re_fit[::-1])
                    selected_indices = np.searchsorted(x_dut_scaled, x_dut_scaled_selected)
                    mask = np.zeros_like(x_dut_scaled, dtype=np.bool)
                    mask[selected_indices] = True
                    plot_utils.plot_alignment_fit(
                        x=x_dut_scaled,
                        mean_fitted=mean_fitted_scaled,
                        mask=mask,
                        fit_fn=fit_fn,
                        fit=re_fit,
                        pcov=re_fit_pcov,
                        chi2=chi2,
                        mean_error_fitted=mean_error_fitted_scaled,
                        n_cluster=n_cluster,
                        n_pixel_ref=n_pixel_ref,
                        n_pixel_dut=n_pixel_dut,
                        pixel_size_ref=pixel_size_ref,
                        pixel_size_dut=pixel_size_dut,
                        ref_name=ref_name,
                        dut_name=dut_name,
                        prefix=table_prefix,
                        output_pdf=output_pdf)

            logging.info('Store pre alignment data in %s', output_alignment_file)
            with tb.open_file(output_alignment_file, mode="w") as out_file_h5:
                try:
                    result_table = out_file_h5.create_table(out_file_h5.root, name='PreAlignment', description=result.dtype, title='Prealignment alignment from correlation', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    result_table.append(result)
                except tb.exceptions.NodeError:
                    logging.warning('Coarse alignment table exists already. Do not create new.')


def _fit_data(x, data, s_n, coeff_fitted, mean_fitted, mean_error_fitted, sigma_fitted, chi2, fit_background, reduce_background):

    def calc_limits_from_fit(coeff):
        ''' Calculates the fit limits from the last successfull fit.'''
        limits = [
            [0.1 * coeff[0], x.min(), 0.5 * coeff[2], 0.01 * coeff[3], x.min(), 0.5 * coeff[5], 0.5 * coeff[6]],
            [10.0 * coeff[0], x.max(), 2.0 * coeff[2], 10.0 * coeff[3], x.max(), 2.0 * coeff[5], 2.0 * coeff[6]]
        ]
        # Fix too small sigma, sigma < 1 is unphysical
        if limits[1][2] < 1.:
            limits[1][2] = 10.
        return limits

    def signal_sanity_check(coeff, signal_noise, A_peak):
        ''' Sanity check if signal was deducted correctly from background.

            3 Conditions:
            1. The given signal to noise value has to be fullfilled: S/N > Amplitude Signal / ( Amplidude background + Offset)
            2. The signal + background has to be large enough: Amplidute 1 + Amplitude 2 + Offset > Data maximum / 2
            3. The Signal Sigma has to be smaller than the background sigma, otherwise beam would be larger than one pixel pitch
        '''
        if coeff[0] < (coeff[3] + coeff[6]) * signal_noise or coeff[0] + coeff[3] + coeff[6] < A_peak / 2.0 or coeff[2] > coeff[5] / 2.0:
            return False
        return True

    n_pixel_dut, n_pixel_ref = data.shape[0], data.shape[1]
    # Start values for fitting
    # Correlation peak
    mu_peak = x[np.argmax(data, axis=1)]
    A_peak = np.max(data, axis=1)  # signal / correlation peak
    # Background of uncorrelated data
    n_entries = np.sum(data, axis=1)
    A_background = np.mean(data, axis=1)  # noise / background halo
    mu_background = np.zeros_like(n_entries)
    mu_background[n_entries > 0] = np.average(data, axis=1, weights=x)[n_entries > 0] * np.sum(x) / n_entries[n_entries > 0]

    coeff = None
    fit_converged = False  # To signal that las fit was good, thus the results can be taken as start values for next fit

    no_correlation_indeces = []
    few_correlation_indeces = []

    for index in np.arange(n_pixel_dut):  # Loop over x dimension of correlation histogram
        # TODO: start fitting from the beam center to get a higher chance to pick up the correlation peak

        # omit correlation fit with no entries / correlation (e.g. sensor edges, masked columns)
        if np.all(data[index, :] == 0):
            no_correlation_indeces.append(index)
            continue

        # omit correlation fit if sum of correlation entries is < 1 % of total entries devided by number of indices
        # (e.g. columns not in the beam)
        n_cluster_curr_index = data[index, :].sum()
        if fit_converged and n_cluster_curr_index < data.sum() / n_pixel_dut * 0.01:
            few_correlation_indeces.append(index)
            continue

        # Set start parameters and fit limits
        # Parameters: A_1, mu_1, sigma_1, A_2, mu_2, sigma_2, offset
        if fit_converged and not reduce_background:  # Set start values from last successfull fit, no large difference expected
            p0 = coeff  # Set start values from last successfull fit
            bounds = calc_limits_from_fit(coeff)  # Set boundaries from previous converged fit
        else:  # No (last) successfull fit, try to dedeuce reasonable start values
            p0 = [A_peak[index], mu_peak[index], 5.0, A_background[index], mu_background[index], analysis_utils.get_rms_from_histogram(data[index, :], x), 0.0]
            bounds = [[0.0, x.min(), 0.0, 0.0, x.min(), 0.0, 0.0], [2.0 * A_peak[index], x.max(), x.max() - x.min(), 2.0 * A_peak[index], x.max(), np.inf, A_peak[index]]]

        # Fit correlation
        if fit_background:  # Describe background with addidional gauss + offset
            try:
                coeff, var_matrix = curve_fit(analysis_utils.double_gauss_offset, x, data[index, :], p0=p0, bounds=bounds)
            except RuntimeError:  # curve_fit failed
                fit_converged = False
            else:
                fit_converged = True
                # do some result checks
                if not signal_sanity_check(coeff, s_n, A_peak[index]):
                    logging.debug('No correlation peak found. Try another fit...')
                    # Use parameters from last fit as start parameters for the refit
                    y_fit = analysis_utils.double_gauss_offset(x, *coeff)
                    try:
                        coeff, var_matrix = refit_advanced(x_data=x, y_data=data[index, :], y_fit=y_fit, p0=coeff)
                    except RuntimeError:  # curve_fit failed
                        fit_converged = False
                    else:
                        fit_converged = True
                        # Check result again:
                        if not signal_sanity_check(coeff, s_n, A_peak[index]):
                            logging.debug('No correlation peak found after refit!')
                            fit_converged = False

        else:  # Describe background with offset only.
            # Change start parameters and boundaries
            p0_gauss_offset = [p0_val for i, p0_val in enumerate(p0) if i in (0, 1, 2, 6)]
            bounds_gauss_offset = [0, np.inf]
            bounds_gauss_offset[0] = [bound_val for i, bound_val in enumerate(bounds[0]) if i in (0, 1, 2, 6)]
            bounds_gauss_offset[1] = [bound_val for i, bound_val in enumerate(bounds[1]) if i in (0, 1, 2, 6)]
            
#             print index
#             print bounds_gauss_offset
#             plt.bar(x, data[index, :], label='Data')

#             xx = np.linspace(x[0], x[-1], num=10000, endpoint=True)
#             plt.plot(xx, analysis_utils.gauss_offset(xx, *p0_gauss_offset), label='Start Values')

            try:
                coeff_gauss_offset, var_matrix = curve_fit(analysis_utils.gauss_offset, x, data[index, :], p0=p0_gauss_offset, bounds=bounds_gauss_offset)
#                 plt.plot(xx, analysis_utils.gauss_offset(xx, *coeff_gauss_offset), label='Fit')
            except RuntimeError:  # curve_fit failed
                fit_converged = False
            else:
                # Correlation should have at least 2 entries to avoid random fluctuation peaks to be selected
                if coeff_gauss_offset[0] > 2:
                    fit_converged = True
                    # Change back coefficents
                    coeff = np.insert(coeff_gauss_offset, 3, [np.nan] * 3)  # Parameters: A_1, mu_1, sigma_1, A_2, mu_2, sigma_2, offset
                else:
                    fit_converged = False
#             plt.grid(True)
#             plt.legend(loc=0)
#             plt.show()
        # Set fit results for given index if successful
        if fit_converged:
            coeff_fitted[index] = coeff
            mean_fitted[index] = coeff[1]
            mean_error_fitted[index] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
            sigma_fitted[index] = np.abs(coeff[2])
            chi2[index] = analysis_utils.get_chi2(y_data=data[index, :], y_fit=analysis_utils.double_gauss_offset(x, *coeff))


    if no_correlation_indeces:
        logging.info('No correlation entries for indeces %s. Omit correlation fit.', str(no_correlation_indeces)[1:-1])

    if few_correlation_indeces:
        logging.info('Very few correlation entries for index %s. Omit correlation fit.', str(few_correlation_indeces)[1:-1])


def refit_advanced(x_data, y_data, y_fit, p0):
    ''' Substract the fit from the data, thus only the small signal peak should be left.
    Fit this peak, and refit everything with start values'''
    y_peak = y_data - y_fit  # Fit most likely only describes background, thus substract it
    peak_A = np.max(y_peak)  # Determine start value for amplitude
    peak_mu = np.argmax(y_peak)  # Determine start value for mu
    fwhm_1, fwhm_2 = analysis_utils.fwhm(x_data, y_peak)
    peak_sigma = (fwhm_2 - fwhm_1) / 2.35  # Determine start value for sigma

    # Fit a Gauss + Offset to the background substracted data
    coeff_peak, _ = curve_fit(analysis_utils.gauss_offset_slope, x_data, y_peak, p0=[peak_A, peak_mu, peak_sigma, 0.0, 0.0], bounds=([0.0, 0.0, 0.0, -10000.0, -10.0], [1.1 * peak_A, np.inf, np.inf, 10000.0, 10.0]))

    # Refit orignial double Gauss function with proper start values for the small signal peak
    coeff, var_matrix = curve_fit(analysis_utils.double_gauss_offset, x_data, y_data, p0=[coeff_peak[0], coeff_peak[1], coeff_peak[2], p0[3], p0[4], p0[5], p0[6]], bounds=[0.0, np.inf])

    return coeff, var_matrix


def apply_alignment(input_hit_file, input_alignment, output_hit_aligned_file, inverse=False, force_prealignment=False, no_z=False, use_duts=None, chunk_size=1000000):
    ''' Takes a file with tables containing hit information (x, y, z) and applies the alignment to each DUT hits positions. The alignment data is used. If this is not
    available a fallback to the pre-alignment is done.
    One can also inverse the alignment or apply the alignment without changing the z position.

    This function cannot be easily made faster with multiprocessing since the computation function (apply_alignment_to_chunk) does not contribute significantly to the runtime (< 20 %),
    but the copy overhead for not shared memory needed for multipgrocessing is higher. Also the hard drive IO can be limiting (30 Mb/s read, 20 Mb/s write to the same disk)

    Parameters
    ----------
    input_hit_file : pytables file
        Input file name with hit data (e.g. merged data file, tracklets file, etc.)
    input_alignment : pytables file or alignment array
        The alignment file with the data
    output_hit_aligned_file : pytables file
        Output file name with hit data after alignment was applied
    inverse : boolean
        Apply the inverse alignment
    force_prealignment : boolean
        Take the pre-alignment, although if a coarse alignment is availale
    no_z : boolean
        Do not change the z alignment. Needed since the z position is special for x / y based plane measurements.
    use_duts : iterable
        Iterable of DUT indices to apply the alignment to. Std. setting is all DUTs.
    chunk_size: int
        Defines the amount of in-RAM data. The higher the more RAM is used and the faster this function works.
    '''
    logging.info('== Apply alignment to %s ==', input_hit_file)

    use_prealignment = True if force_prealignment else False

    try:
        with tb.open_file(input_alignment, mode="r") as in_file_h5:  # Open file with alignment data
            alignment = in_file_h5.root.PreAlignment[:]
            if not use_prealignment:
                try:
                    alignment = in_file_h5.root.Alignment[:]
                    logging.info('Use alignment data from file')
                except tb.exceptions.NodeError:
                    use_prealignment = True
                    logging.info('Use pre-alignment data from file')
    except TypeError:  # The input_alignment is an array
        alignment = input_alignment
        try:  # Check if array is prealignent array
            alignment['column_c0']
            logging.info('Use pre-alignment data')
            use_prealignment = True
        except ValueError:
            logging.info('Use alignment data')
            use_prealignment = False

    n_duts = alignment.shape[0]

    def apply_alignment_to_chunk(hits_chunk, dut_index, alignment, inverse, no_z):
        selection = hits_chunk['x_dut_%d' % dut_index] != np.nan  # Do not change virtual hits

        if use_prealignment:  # Apply transformation from pre-alignment information
            hits_chunk['x_dut_%d' % dut_index][selection], hits_chunk['y_dut_%d' % dut_index][selection], hit_z = geometry_utils.apply_alignment(hits_x=hits_chunk['x_dut_%d' % dut_index][selection],
                                                                                                                                                 hits_y=hits_chunk['y_dut_%d' % dut_index][selection],
                                                                                                                                                 hits_z=hits_chunk['z_dut_%d' % dut_index][selection],
                                                                                                                                                 dut_index=dut_index,
                                                                                                                                                 prealignment=alignment,
                                                                                                                                                 inverse=inverse)
        else:  # Apply transformation from fine alignment information
            hits_chunk['x_dut_%d' % dut_index][selection], hits_chunk['y_dut_%d' % dut_index][selection], hit_z = geometry_utils.apply_alignment(hits_x=hits_chunk['x_dut_%d' % dut_index][selection],
                                                                                                                                                 hits_y=hits_chunk['y_dut_%d' % dut_index][selection],
                                                                                                                                                 hits_z=hits_chunk['z_dut_%d' % dut_index][selection],
                                                                                                                                                 dut_index=dut_index,
                                                                                                                                                 alignment=alignment,
                                                                                                                                                 inverse=inverse)
        if not no_z:
            hits_chunk['z_dut_%d' % dut_index][selection] = hit_z

    # Looper over the hits of all DUTs of all hit tables in chunks and apply the alignment
    with tb.open_file(input_hit_file, mode='r') as in_file_h5:
        with tb.open_file(output_hit_aligned_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:  # Loop over potential hit tables in data file
                hits = node
                new_node_name = hits.name

                if new_node_name == 'MergedCluster':  # Merged cluster with alignment are tracklets
                    new_node_name = 'Tracklets'

                hits_aligned_table = out_file_h5.create_table(out_file_h5.root, name=new_node_name, description=np.zeros((1,), dtype=hits.dtype).dtype, title=hits.title, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=hits.shape[0], term_width=80)
                progress_bar.start()

                for hits_chunk, index in analysis_utils.data_aligned_at_events(hits, chunk_size=chunk_size):  # Loop over the hits
                    for dut_index in range(0, n_duts):  # Loop over the DUTs in the hit table
                        if use_duts is not None and dut_index not in use_duts:  # omit DUT
                            continue

                        apply_alignment_to_chunk(hits_chunk, dut_index, alignment, inverse, no_z)

                    hits_aligned_table.append(hits_chunk)
                    progress_bar.update(index)
                progress_bar.finish()

    logging.debug('File with newly aligned hits %s', output_hit_aligned_file)


def alignment(input_track_candidates_file, input_alignment_file, n_pixels, pixel_size, align_duts=None, selection_fit_duts=None, selection_hit_duts=None, selection_track_quality=None, initial_rotation=None, initial_translation=None, max_iterations=10, use_n_tracks=200000, plot_result=True, chunk_size=100000):
    ''' This function does an alignment of the DUTs and sets translation and rotation values for all DUTs.
    The reference DUT defines the global coordinate system position at 0, 0, 0 and should be well in the beam and not heavily rotated.

    To solve the chicken-and-egg problem that a good dut alignment needs hits belonging to one track, but good track finding needs a good dut alignment this
    function work only on already prealigned hits belonging to one track. Thus this function can be called only after track finding.

    These steps are done
    1. Take the found tracks and revert the pre-alignment
    2. Take the track hits belonging to one track and fit tracks for all DUTs
    3. Calculate the residuals for each DUT
    4. Deduce rotations from the residuals and apply them to the hits
    5. Deduce the translation of each plane
    6. Store and apply the new alignment

    repeat step 3 - 6 until the total residual does not decrease (RMS_total = sqrt(RMS_x_1^2 + RMS_y_1^2 + RMS_x_2^2 + RMS_y_2^2 + ...))

    Parameters
    ----------
    input_track_candidates_file : string
        file name with the track candidates table
    input_alignment_file : pytables file
        File name of the input aligment data
    n_pixels : iterable of tuples
        The number of pixels per DUT in (column, row).
        E.g.: two DUTS: [(80, 336), (80, 336)]
    pixel_size : iterable of tuples
        The pixel sizer per DUT in (column, row) in um.
        E.g.: two DUTS: [(50, 250), (50, 250)]
    align_duts : iterable or iterable of iterable
        The combination of duts that are algined at once. One should always align the high resolution planes first.
        E.g. for a telesope (first and last 3 planes) with 2 devices in the center (3, 4):
        align_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
                    [4],  # Align first DUT
                    [3]],  # Align second DUT
    selection_fit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        selection_fit_duts=[0, 1, 2, 5, 6, 7]
    selection_hit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        selection_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    max_iterations : int
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations)
    selection_hit_duts : iterable, or iterable of iterable
        The duts that are required to have a hit with the given track quality. Otherwise the track is omitted
        If None: require all DUTs to have a hit, but if require_dut_hit = False do not use actual fit_dut.
        If iterable: use selection for all dut selections
        If iterable of iterable: define dut with hits for each dut selection separately.
        E.g: for 2 devices: selection_hit_duts = ((1, 2), (0, 1, 2))
    use_n_tracks: int
        Defines the amount of tracks to be used for the alignment. More tracks can potentially make the result
        more precise, but will also increase the calculation time.
    plot_result : boolean
        If true the final alignment applied to the complete data set is plotted. If you have hugh amount
        of data, deactivate this to save time.
    chunk_size: int
        Defines the amount of in-RAM data. The higher the more RAM is used and the faster this function works.
    '''

    logging.info('=== Aligning DUTs ===')

    def calculate_translation_alignment(track_candidates_file, fit_duts, n_duts, selection_fit_duts, selection_hit_duts, selection_track_quality, max_iterations, output_pdf, plot_title_prefix=''):
        ''' Main function that fits tracks, calculates the residuals, deduces rotation and translation values from the residuals
        and applies the new alignment to the track hits. The alignment result is scored as a combined
        residual value of all planes that are being aligned in x and y weighted by the pixel pitch in x and y. '''
        with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
            alignment_last_iteration = in_file_h5.root.Alignment[:]

        total_residual = None
        for iteration in range(max_iterations):
            if iteration >= max_iterations:
                raise RuntimeError('Did not converge to good solution in %d iterations. Increase max_iterations', iteration)

            apply_alignment(input_hit_file=track_candidates_file,  # Always apply alignment to starting file
                            input_alignment=input_alignment_file,
                            output_hit_aligned_file=track_candidates_file[:-3] + '_no_align_%d_tmp.h5' % iteration,
                            inverse=False,
                            force_prealignment=False,
                            chunk_size=chunk_size)

            # Step 2: Fit tracks for all DUTs
            logging.info('= Alignment step 2 / iteration %d: Fit tracks for all DUTs =', iteration)
            fit_tracks(input_track_candidates_file=track_candidates_file[:-3] + '_no_align_%d_tmp.h5' % iteration,
                       input_alignment_file=input_alignment_file,
                       output_tracks_file=track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % iteration,
                       fit_duts=fit_duts,  # Only create residuals of selected DUTs
                       selection_fit_duts=selection_fit_duts,   # Only use selected DUTs for track fit
                       selection_hit_duts=selection_hit_duts,  # Only use selected duts
                       exclude_dut_hit=False,  # For constrained residuals
                       selection_track_quality=selection_track_quality,
                       force_prealignment=False)

            # Step 3: Calculate the residuals for each DUT
            logging.info('= Alignment step 3 / iteration %d: Calculate the residuals for each selected DUT =', iteration)
            calculate_residuals(input_tracks_file=track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % iteration,
                                input_alignment_file=input_alignment_file,
                                output_residuals_file=track_candidates_file[:-3] + '_residuals_%d_tmp.h5' % iteration,
                                n_pixels=n_pixels,
                                pixel_size=pixel_size,
                                output_pdf=False,
                                chunk_size=chunk_size,
                                npixels_per_bin=5 if (iteration in [0, 1, 2]) else None,  # use a coarse binning for the first steps, FIXME: good code practice: nothing hardcoded
                                nbins_per_pixel=1 if (iteration in [0, 1, 2]) else None)  # use a coarse binning for the first steps, FIXME: good code practice: nothing hardcoded

            # Step 4: Deduce rotations from the residuals
            logging.info('= Alignment step 4 / iteration %d: Deduce rotations and translations from the residuals =', iteration)
            alignment_parameters_change, new_total_residual = _analyze_residuals(residuals_file_h5=track_candidates_file[:-3] + '_residuals_%d_tmp.h5' % iteration,
                                                                                 output_fig=output_pdf,
                                                                                 fit_duts=fit_duts,
                                                                                 pixel_size=pixel_size,
                                                                                 n_duts=n_duts,
                                                                                 translation_only=False,
                                                                                 plot_title_prefix=plot_title_prefix,
                                                                                 relaxation_factor=1.0)  # FIXME: good code practice: nothing hardcoded

            # Create actual alignment (old alignment + the actual relative change)
            new_alignment_parameters = geometry_utils.merge_alignment_parameters(
                alignment_last_iteration,
                alignment_parameters_change,
                mode='relative')

# FIXME: This step does not work well
#             # Step 5: Try to find better rotation by minimizing the residual in x + y for different angles
#             logging.info('= Alignment step 5 / iteration %d: Optimize alignment by minimizing residuals =', iteration)
#             new_alignment_parameters, new_total_residual = _optimize_alignment(input_tracks_file=track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % iteration,
#                                                                                alignment_last_iteration=alignment_last_iteration,
#                                                                                new_alignment_parameters=new_alignment_parameters,
#                                                                                pixel_size=pixel_size)

            # Delete not needed files
            os.remove(track_candidates_file[:-3] + '_no_align_%d_tmp.h5' % iteration)
            os.remove(track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % iteration)
            os.remove(track_candidates_file[:-3] + '_tracks_%d_tmp.pdf' % iteration)
            os.remove(track_candidates_file[:-3] + '_residuals_%d_tmp.h5' % iteration)
            logging.info('Total residual %1.4e', new_total_residual)

            if total_residual is not None and new_total_residual > total_residual:  # True if actual alignment is worse than the alignment from last iteration
                logging.info('!! Best alignment found !!')
                logging.info('= Alignment step 6 / iteration %d: Use rotation / translation information from previous iteration =', iteration)
                geometry_utils.store_alignment_parameters(input_alignment_file,  # Store alignment from last iteration
                                                          alignment_last_iteration,
                                                          mode='absolute',
                                                          select_duts=fit_duts)
                return
            else:
                total_residual = new_total_residual

# with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
            alignment_last_iteration = new_alignment_parameters.copy()  # in_file_h5.root.Alignment[:]

            logging.info('= Alignment step 6 / iteration %d: Set new rotation / translation information in alignment file =', iteration)
            geometry_utils.store_alignment_parameters(input_alignment_file,
                                                      new_alignment_parameters,
                                                      mode='absolute',
                                                      select_duts=fit_duts)

    def duts_alignment(align_duts, n_duts, selection_fit_duts, selection_hit_duts, selection_track_quality, alignment_index):  # Called for each list of DUTs to align

        # Step 0: Reduce the number of tracks to increase the calculation time
        logging.info('= Alignment step 0: Reduce number of tracks to %d =', use_n_tracks)
        track_quality_mask = 0
        for index, dut in enumerate(selection_hit_duts):
            for quality in range(3):
                if quality <= selection_track_quality[index]:
                    track_quality_mask |= ((1 << dut) << quality * 8)

        logging.info('Use track with hits in DUTs %s', str(selection_hit_duts)[1:-1])
        data_selection.select_hits(hit_file=input_track_candidates_file,
                                   output_file=input_track_candidates_file[:-3] + '_reduced_%d.h5' % alignment_index,
                                   max_hits=use_n_tracks,
                                   track_quality=track_quality_mask,
                                   track_quality_mask=track_quality_mask,
                                   chunk_size=chunk_size)
        input_track_candidates_reduced = input_track_candidates_file[:-3] + '_reduced_%d.h5' % alignment_index

        # Step 1: Take the found tracks and revert the pre-alignment to start alignment from the beginning
        logging.info('= Alignment step 1: Revert pre-alignment =')
        apply_alignment(input_hit_file=input_track_candidates_reduced,
                        input_alignment=input_alignment_file,  # Revert prealignent
                        output_hit_aligned_file=input_track_candidates_reduced[:-3] + '_not_aligned.h5',
                        inverse=True,
                        force_prealignment=True,
                        chunk_size=chunk_size)

        # Stage N: Repeat alignment with constrained residuals until total residual does not decrease anymore
        calculate_translation_alignment(track_candidates_file=input_track_candidates_reduced[:-3] + '_not_aligned.h5',
                                        fit_duts=align_duts,  # Only use the actual DUTs to align
                                        n_duts=n_duts,
                                        selection_fit_duts=selection_fit_duts,
                                        selection_hit_duts=selection_hit_duts,
                                        selection_track_quality=selection_track_quality,
                                        max_iterations=max_iterations,
                                        output_pdf=False)

        # Plot final result
        if plot_result:
            logging.info('= Alignment step 7: Plot final result =')
            with PdfPages(os.path.join(os.path.dirname(os.path.realpath(input_track_candidates_file)), 'Alignment_%d.pdf' % alignment_index)) as output_pdf:
                # Apply final alignment result
                apply_alignment(input_hit_file=input_track_candidates_reduced[:-3] + '_not_aligned.h5',
                                input_alignment=input_alignment_file,
                                output_hit_aligned_file=input_track_candidates_file[:-3] + '_final_tmp_%d.h5' % alignment_index,
                                chunk_size=chunk_size)
                fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_final_tmp_%d.h5' % alignment_index,
                           input_alignment_file=input_alignment_file,
                           output_tracks_file=input_track_candidates_file[:-3] + '_tracks_final_tmp_%d.h5' % alignment_index,
                           fit_duts=align_duts,  # Only create residuals of selected DUTs
                           selection_fit_duts=selection_fit_duts,  # Only use selected duts
                           selection_hit_duts=selection_hit_duts,
                           exclude_dut_hit=True,  # For unconstrained residuals
                           selection_track_quality=selection_track_quality)
                calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_final_tmp_%d.h5' % alignment_index,
                                    input_alignment_file=input_alignment_file,
                                    output_residuals_file=input_track_candidates_file[:-3] + '_residuals_final_tmp_%d.h5' % alignment_index,
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    output_pdf=output_pdf,
                                    chunk_size=chunk_size)
                os.remove(input_track_candidates_file[:-3] + '_final_tmp_%d.h5' % alignment_index)
                os.remove(input_track_candidates_file[:-3] + '_tracks_final_tmp_%d.h5' % alignment_index)
                os.remove(input_track_candidates_file[:-3] + '_tracks_final_tmp_%d.pdf' % alignment_index)
                os.remove(input_track_candidates_file[:-3] + '_residuals_final_tmp_%d.h5' % alignment_index)
        os.remove(input_track_candidates_reduced[:-3] + '_not_aligned.h5')
        os.remove(input_track_candidates_file[:-3] + '_reduced_%d.h5' % alignment_index)

    # Open the pre-alignment and create empty alignment info (at the beginning only the z position is set)
    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        alignment_parameters = _create_alignment_array(n_duts)
        alignment_parameters['translation_z'] = prealignment['z']

        if initial_rotation:
            if isinstance(initial_rotation[0], Iterable):
                for dut_index in range(n_duts):
                    alignment_parameters['alpha'][dut_index] = initial_rotation[dut_index][0]
                    alignment_parameters['beta'][dut_index] = initial_rotation[dut_index][1]
                    alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
            else:
                for dut_index in range(n_duts):
                    alignment_parameters['alpha'][dut_index] = initial_rotation[0]
                    alignment_parameters['beta'][dut_index] = initial_rotation[1]
                    alignment_parameters['gamma'][dut_index] = initial_rotation[2]

        if initial_translation:
            if isinstance(initial_translation[0], Iterable):
                for dut_index in range(n_duts):
                    alignment_parameters['translation_x'][dut_index] = initial_translation[dut_index][0]
                    alignment_parameters['translation_y'][dut_index] = initial_translation[dut_index][1]
            else:
                for dut_index in range(n_duts):
                    alignment_parameters['translation_x'][dut_index] = initial_translation[0]
                    alignment_parameters['translation_y'][dut_index] = initial_translation[1]

        if np.any(np.abs(alignment_parameters['alpha']) > np.pi / 4.) or np.any(np.abs(alignment_parameters['beta']) > np.pi / 4.) or np.any(np.abs(alignment_parameters['gamma']) > np.pi / 4.):
            logging.warning('A rotation angle > pi / 4 is not supported, you should set the correct angle and translation as a start parameter, sorry!')

    geometry_utils.store_alignment_parameters(input_alignment_file,
                                              alignment_parameters=alignment_parameters,
                                              mode='absolute')

    # Create list with combinations of DUTs to align
    if align_duts is None:  # Align all duts
        align_duts = [range(n_duts)]
    elif not isinstance(align_duts[0], Iterable):
        align_duts = [align_duts]

    # Check if some DUTs are not aligned
    missing_duts = []
    for dut in range(n_duts):
        if not any([i == dut for j in align_duts for i in j]):
            missing_duts.append(dut)
    if len(missing_duts) > 0:
        logging.warning('These DUTs will not be aligned: %s', str(missing_duts)[1:-1])

    # Loop over all combinations of DUTs to align, simplest case: use all DUTs at once to align
    # Usual case: align high resolution devices first, then other devices
    for index, actual_align_duts in enumerate(align_duts):
        if not selection_fit_duts:
            actual_selection_fit_duts = actual_align_duts
        elif isinstance(selection_fit_duts[index], Iterable):
            actual_selection_fit_duts = selection_fit_duts[index]
        else:
            actual_selection_fit_duts = selection_fit_duts
        if not selection_hit_duts:
            actual_selection_hit_duts = actual_align_duts
        elif isinstance(selection_hit_duts[index], Iterable):
            actual_selection_hit_duts = selection_hit_duts[index]
        else:
            actual_selection_hit_duts = selection_hit_duts
        if not selection_track_quality:
            actual_selection_track_quality = [1] * len(actual_selection_hit_duts)
        elif isinstance(selection_track_quality[index], Iterable):
            actual_selection_track_quality = selection_track_quality[index]
        else:
            actual_selection_track_quality = selection_track_quality

        logging.info('Align DUTs %s', str(actual_align_duts)[1:-1])

        duts_alignment(align_duts=actual_align_duts, n_duts=n_duts, selection_fit_duts=actual_selection_fit_duts, selection_hit_duts=actual_selection_hit_duts, selection_track_quality=actual_selection_track_quality, alignment_index=index)

    logging.info('Alignment finished successfully!')


# Helper functions for the alignment. Not to be used directly.
def _create_alignment_array(n_duts):
    # Result Translation / rotation table
    description = [('DUT', np.int)]
    description.append(('translation_x', np.float))
    description.append(('translation_y', np.float))
    description.append(('translation_z', np.float))
    description.append(('alpha', np.float))
    description.append(('beta', np.float))
    description.append(('gamma', np.float))
    description.append(('correlation_x', np.float))
    description.append(('correlation_y', np.float))

    array = np.zeros((n_duts,), dtype=description)
    array[:]['DUT'] = np.array(range(n_duts))
    return array


def _analyze_residuals(residuals_file_h5, output_fig, fit_duts, pixel_size, n_duts, translation_only=False, relaxation_factor=1.0, plot_title_prefix=''):
    ''' Take the residual plots and deduce rotation and translation angles from them '''
    alignment_parameters = _create_alignment_array(n_duts)

    total_residual = 0  # Sum of all residuals to judge the overall alignment

    with tb.open_file(residuals_file_h5) as in_file_h5:
        for dut_index in fit_duts:
            alignment_parameters[dut_index]['DUT'] = dut_index
            # Global residuals
            hist_node = in_file_h5.get_node('/ResidualsX_DUT%d' % dut_index)
            std_x = hist_node._v_attrs.fit_coeff[2]

            # Add resdidual to total residual normalized to pixel pitch in x
            total_residual = np.sqrt(np.square(total_residual) + np.square(std_x / pixel_size[dut_index][0]))

            if output_fig:
                plot_utils.plot_residuals(histogram=hist_node[:],
                                          edges=hist_node._v_attrs.xedges,
                                          fit=hist_node._v_attrs.fit_coeff,
                                          fit_errors=hist_node._v_attrs.fit_cov,
                                          title='Residuals for DUT %d' % dut_index,
                                          x_label='X residual [um]',
                                          output_fig=output_fig)

            hist_node = in_file_h5.get_node('/ResidualsY_DUT%d' % dut_index)
            std_y = hist_node._v_attrs.fit_coeff[2]

            # Add resdidual to total residual normalized to pixel pitch in y
            total_residual = np.sqrt(np.square(total_residual) + np.square(std_y / pixel_size[dut_index][1]))

            if translation_only:
                return alignment_parameters, total_residual

            if output_fig:
                plot_utils.plot_residuals(histogram=hist_node[:],
                                          edges=hist_node._v_attrs.xedges,
                                          fit=hist_node._v_attrs.fit_coeff,
                                          fit_errors=hist_node._v_attrs.fit_cov,
                                          title='Residuals for DUT %d' % dut_index,
                                          x_label='Y residual [um]',
                                          output_fig=output_fig)

            # use offset at origin of sensor (center of sensor) to calculate x and y correction
            # do not use mean/median of 1D residual since it depends on the beam spot position when the device is rotated
            mu_x = in_file_h5.get_node_attr('/YResidualsX_DUT%d' % dut_index, 'fit_coeff')[0]
            mu_y = in_file_h5.get_node_attr('/XResidualsY_DUT%d' % dut_index, 'fit_coeff')[0]
            # use slope to calculate alpha, beta and gamma
            m_xx = in_file_h5.get_node_attr('/XResidualsX_DUT%d' % dut_index, 'fit_coeff')[1]
            m_yy = in_file_h5.get_node_attr('/YResidualsY_DUT%d' % dut_index, 'fit_coeff')[1]
            m_xy = in_file_h5.get_node_attr('/XResidualsY_DUT%d' % dut_index, 'fit_coeff')[1]
            m_yx = in_file_h5.get_node_attr('/YResidualsX_DUT%d' % dut_index, 'fit_coeff')[1]

            alpha, beta, gamma = analysis_utils.get_rotation_from_residual_fit(m_xx=m_xx, m_xy=m_xy, m_yx=m_yx, m_yy=m_yy)

            alignment_parameters[dut_index]['correlation_x'] = std_x
            alignment_parameters[dut_index]['translation_x'] = -mu_x
            alignment_parameters[dut_index]['correlation_y'] = std_y
            alignment_parameters[dut_index]['translation_y'] = -mu_y
            alignment_parameters[dut_index]['alpha'] = alpha * relaxation_factor
            alignment_parameters[dut_index]['beta'] = beta * relaxation_factor
            alignment_parameters[dut_index]['gamma'] = gamma * relaxation_factor

    return alignment_parameters, total_residual


def _optimize_alignment(input_tracks_file, alignment_last_iteration, new_alignment_parameters, pixel_size):
    ''' Changes the angles of a virtual plane such that the projected track intersections onto this virtual plane
    are most close to the measured hits on the real DUT at this position. Then the angles of the virtual plane
    should correspond to the real DUT angles. The distance is not weighted quadratically (RMS) but linearly since
    this leads to better results (most likely heavily scattered tracks / beam angle spread at the edges are weighted less).'''
    # Create new absolute alignment
    alignment_result = new_alignment_parameters

    def _minimize_me(align, dut_position, hit_x_local, hit_y_local, hit_z_local, pixel_size, offsets, slopes):
        # Calculate intersections with a dut plane given by alpha, beta, gamma at the dut_position in the global coordinate system
        rotation_matrix = geometry_utils.rotation_matrix(alpha=align[0],
                                                         beta=align[1],
                                                         gamma=align[2])
        basis_global = rotation_matrix.T.dot(np.eye(3))
        dut_plane_normal = basis_global[2]
        actual_dut_position = dut_position.copy()
        actual_dut_position[2] = align[3] * 1e6  # Convert z position from m to um
        intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                         line_directions=slopes,
                                                                         position_plane=actual_dut_position,
                                                                         normal_plane=dut_plane_normal)

        # Transform to the local coordinate system to compare with measured hits
        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=actual_dut_position[0],
                                                                                     y=actual_dut_position[1],
                                                                                     z=actual_dut_position[2],
                                                                                     alpha=align[0],
                                                                                     beta=align[1],
                                                                                     gamma=align[2])

        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_transformation_matrix(x=intersections[:, 0],
                                                                                                                      y=intersections[:, 1],
                                                                                                                      z=intersections[:, 2],
                                                                                                                      transformation_matrix=transformation_matrix)

        # Cross check if transformations are correct (z == 0 in the local coordinate system)
        if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0) or not np.allclose(intersection_z_local, 0):
            logging.error('Hit z position = %s and z intersection %s',
                          str(hit_z_local[~np.isclose(hit_z_local, 0)][:3]),
                          str(intersection_z_local[~np.isclose(intersection_z_local, 0)][:3]))
            raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

        return np.sum(np.abs(hit_x_local - intersection_x_local) / pixel_size[0]) + np.sum(np.abs(hit_y_local - intersection_y_local)) / pixel_size[1]
#         return np.sqrt(np.square(np.std(hit_x_local - intersection_x_local) / pixel_size[0]) + np.square(np.std(hit_y_local - intersection_y_local)) / pixel_size[1])

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        residuals_before = []
        residuals_after = []
        for node in in_file_h5.root:
            actual_dut = int(re.findall(r'\d+', node.name)[-1])
            dut_position = np.array([alignment_last_iteration[actual_dut]['translation_x'], alignment_last_iteration[actual_dut]['translation_y'], alignment_last_iteration[actual_dut]['translation_z']])

            # Hits with the actual alignment
            hits = np.vstack((node[:]['x_dut_%d' % actual_dut], node[:]['y_dut_%d' % actual_dut], node[:]['z_dut_%d' % actual_dut])).T

            # Transform hits to the local coordinate system
            hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hits_x=hits[:, 0],
                                                                                   hits_y=hits[:, 1],
                                                                                   hits_z=hits[:, 2],
                                                                                   dut_index=actual_dut,
                                                                                   alignment=alignment_last_iteration,
                                                                                   inverse=True)

            # Track infos
            offsets = np.vstack((node[:]['offset_0'], node[:]['offset_1'], node[:]['offset_2'])).T
            slopes = np.vstack((node[:]['slope_0'], node[:]['slope_1'], node[:]['slope_2'])).T

            # Rotation start values of minimizer
            alpha = alignment_result[actual_dut]['alpha']
            beta = alignment_result[actual_dut]['beta']
            gamma = alignment_result[actual_dut]['gamma']
            z_position = alignment_result[actual_dut]['translation_z']

            # Trick to have the same order of magnitue of variation for angles and position, otherwise scipy minimizers
            # do not converge if step size of parameters is very different
            z_position_in_m = z_position / 1e6

            residual = _minimize_me(np.array([alpha, beta, gamma, z_position_in_m]),
                                    dut_position,
                                    hit_x_local,
                                    hit_y_local,
                                    hit_z_local,
                                    pixel_size[actual_dut],
                                    offsets,
                                    slopes)
            residuals_before.append(residual)
            logging.info('Optimize angles / z of DUT %d with start parameters: %1.2e, %1.2e, %1.2e Rad and z = %d um with residual %1.2e' % (actual_dut,
                                                                                                                                             alpha,
                                                                                                                                             beta,
                                                                                                                                             gamma,
                                                                                                                                             z_position_in_m * 1e6,
                                                                                                                                             residual))

            # FIXME:
            # Has to be heavily restricted otherwise converges to unphysical solutions since the scoring with residuals is not really working well
            bounds = [(alpha - 0.01, alpha + 0.01), (beta - 0.01, beta + 0.01), (gamma - 0.001, gamma + 0.001), (z_position_in_m - 10e-6, z_position_in_m + 10e-6)]

            result = minimize(fun=_minimize_me,
                              x0=np.array([alpha, beta, gamma, z_position_in_m]),  # Start values from residual fit
                              args=(dut_position, hit_x_local, hit_y_local, hit_z_local, pixel_size[actual_dut], offsets, slopes),
                              bounds=bounds,
                              method='SLSQP')

            alpha, beta, gamma, z_position_in_m = result.x
            residual = _minimize_me(result.x,
                                    dut_position,
                                    hit_x_local,
                                    hit_y_local,
                                    hit_z_local,
                                    pixel_size[actual_dut],
                                    offsets,
                                    slopes)
            residuals_after.append(residual)

            logging.info('Found angles of DUT %d with best angles: %1.2e, %1.2e, %1.2e Rad and z = %d um with residual %1.2e' % (actual_dut,
                                                                                                                                 alpha,
                                                                                                                                 beta,
                                                                                                                                 gamma,
                                                                                                                                 z_position_in_m * 1e6,
                                                                                                                                 residual))
            # Rotation start values of minimizer
            alignment_result[actual_dut]['alpha'] = alpha
            alignment_result[actual_dut]['beta'] = beta
            alignment_result[actual_dut]['gamma'] = gamma
            alignment_result[actual_dut]['translation_z'] = z_position_in_m * 1e6  # convert z position from m to um

    total_residuals_before = np.sqrt(np.sum(np.square(np.array(residuals_before))))
    total_residuals_after = np.sqrt(np.sum(np.square(np.array(residuals_after))))
    logging.info('Reduced the total residuals in the optimization steps from %1.2e to %1.2e', total_residuals_before, total_residuals_after)
    if total_residuals_before < total_residuals_after:
        raise RuntimeError('Alignment optimization did not converge!')

    return alignment_result, total_residuals_after  # Return alignment result and total residual


# Helper functions to be called from multiple processes
def _correlate_cluster(cluster_dut_0, cluster_file, start_index, start_event_number, stop_event_number, column_correlation, row_correlation, chunk_size):
    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
        for actual_dut_cluster, start_index in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start=start_index, start_event_number=start_event_number, stop_event_number=stop_event_number, chunk_size=chunk_size):  # Loop over the cluster in the actual cluster file in chunks

            analysis_utils.correlate_cluster_on_event_number(data_1=cluster_dut_0,
                                                             data_2=actual_dut_cluster,
                                                             column_corr_hist=column_correlation,
                                                             row_corr_hist=row_correlation)

    return start_index, column_correlation, row_correlation
