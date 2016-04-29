''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import re
import progressbar
import warnings
import os.path

import matplotlib.pyplot as plt
import tables as tb
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, leastsq, basinhopping, OptimizeWarning
from matplotlib.backends.backend_pdf import PdfPages

from multiprocessing import Pool, cpu_count
# from math import sqrt
# from math import asin

from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils

# Imports for track based alignment
from testbeam_analysis.track_analysis import fit_tracks
from testbeam_analysis.result_analysis import calculate_residuals
from pybar.analysis.analysis_utils import InvalidInputError

warnings.simplefilter("ignore", OptimizeWarning)  # Fit errors are handled internally, turn of warnings


def correlate_cluster(input_cluster_files, output_correlation_file, n_pixels, pixel_size, dut_names=None, output_pdf_file=None, chunk_size=4999999):
    '''Histograms the cluster mean column (row) of two different devices on an event basis. If the cluster means are correlated a line should be seen.
    The cluster means are round to 1 um precision to increase the histogramming speed.
    All permutations are considered (all cluster of the first device are correlated with all cluster of the second device).

    Parameters
    ----------
    input_cluster_files : iterable of pytables file
        Input files with cluster data. One file per DUT.
    output_correlation_file : pytables file
        Output file with the correlation histograms.
    n_pixel : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixel = [(80, 336), (80, 336)]
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension in um in column, row direction
        e.g. for 2 DUTs: pixel_size = [(250, 50), (250, 50)]
    dut_names : iterable of strings
        To show the DUT names in the plot
    output_pdf_file : string
        File name for the output plots.
    chunk_size: int
        Defines the amount of in-RAM data. The higher the more RAM is used and the faster this function works.
    '''

    logging.info('=== Correlate the position of %d DUTs ===', len(input_cluster_files))
    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        n_duts = len(input_cluster_files)

        # Result arrays to be filled
        column_correlations = [None] * (n_duts - 1)
        row_correlations = [None] * (n_duts - 1)

        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Cluster.shape[0], term_width=80)
            progress_bar.start()
            start_indices = [0] * (n_duts - 1)  # Store the loop indices for speed up
            for cluster_dut_0, index in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = cluster_dut_0[:]['event_number']
                # Calculate the common event number of each device with the reference device and correlate the cluster of this events
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_dut_cluster, start_indices[dut_index - 1] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start=start_indices[dut_index - 1], start_event_number=actual_event_numbers[0], stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size):  # Loop over the cluster in the actual cluster file in chunks
                            dut_0_cluster, actual_dut_cluster = analysis_utils.merge_on_event_number(cluster_dut_0, actual_dut_cluster)

                            # Convert to integer to allow fast histogramming
                            dut0_mean_cluster = (dut_0_cluster['mean_column'] - 1.).astype(np.int)
                            dut0_mean_row = (dut_0_cluster['mean_row'] - 1.).astype(np.int)
                            actual_dut_mean_cluster = (actual_dut_cluster['mean_column'] - 1.).astype(np.int)
                            actual_dut_mean_row = (actual_dut_cluster['mean_row'] - 1.).astype(np.int)
                            shape_column = (n_pixels[dut_index][0], n_pixels[0][0])
                            shape_row = (n_pixels[dut_index][1], n_pixels[0][1])

                            if not np.any(column_correlations[dut_index - 1]):
                                column_correlations[dut_index - 1] = analysis_utils.hist_2d_index(actual_dut_mean_cluster, dut0_mean_cluster, shape=shape_column)
                                row_correlations[dut_index - 1] = analysis_utils.hist_2d_index(actual_dut_mean_row, dut0_mean_row, shape=shape_row)
                            else:
                                column_correlations[dut_index - 1] += analysis_utils.hist_2d_index(actual_dut_mean_cluster, dut0_mean_cluster, shape=shape_column)
                                row_correlations[dut_index - 1] += analysis_utils.hist_2d_index(actual_dut_mean_row, dut0_mean_row, shape=shape_row)

                progress_bar.update(index)

            # Store the correlation histograms
            for dut_index in range(n_duts - 1):
                out_col = out_file_h5.createCArray(out_file_h5.root, name='CorrelationColumn_%d_0' % (dut_index + 1), title='Column Correlation between DUT %d and %d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(column_correlations[dut_index].dtype), shape=column_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row = out_file_h5.createCArray(out_file_h5.root, name='CorrelationRow_%d_0' % (dut_index + 1), title='Row Correlation between DUT %d and %d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(row_correlations[dut_index].dtype), shape=row_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col.attrs.filenames = [str(input_cluster_files[0]), str(input_cluster_files[dut_index])]
                out_row.attrs.filenames = [str(input_cluster_files[0]), str(input_cluster_files[dut_index])]
                out_col[:] = column_correlations[dut_index]
                out_row[:] = row_correlations[dut_index]
            progress_bar.finish()

    plot_utils.plot_correlations(input_correlation_file=output_correlation_file, pixel_size=pixel_size, dut_names=dut_names)


def prealignment(input_correlation_file, output_alignment_file, z_positions, pixel_size, dut_names=None, non_interactive=True, iterations=3, fix_slope=False):
    '''Deduce a prealignment from the correlations, by fitting the correlations with a straight line (gives offset, slope, but no tild angles).
       The user can define cuts on the fit error and straight line offset in an interactive way.

        Parameters
    ----------
    input_correlation_file : pytbales file
        The input file with the correlation histograms.
    output_alignment_file : pytables file
        The output file for correlation data.
    z_positions : iterable
        The positions of the devices in z in um
    pixel_size: iterable
        Iterable of tuples with column and row pixel size in um
    dut_names: iterable
        List of names of the DUTs.
    non_interactive : boolean
        Deactivate user interaction and apply cuts automatically
    iterations : number
        Only used in non interactive mode. Sets how often automatic cuts are applied.
    '''

    logging.info('=== Prealignment ===')

    def gauss_offset(x, *p):
        A, mu, sigma, offset, slope = p
        return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0)) + offset + x * slope

    with PdfPages('Prealignment.pdf') as output_pdf:
        with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
            n_nodes = len(in_file_h5.list_nodes("/")) // 2 + 1
            result = np.zeros(shape=(n_nodes,), dtype=[('DUT', np.uint8), ('column_c0', np.float), ('column_c0_error', np.float), ('column_c1', np.float), ('column_c1_error', np.float), ('column_sigma', np.float), ('column_sigma_error', np.float), ('row_c0', np.float), ('row_c0_error', np.float), ('row_c1', np.float), ('row_c1_error', np.float), ('row_sigma', np.float), ('row_sigma_error', np.float), ('z', np.float)])
            # Set std. settings for reference DUT0
            result[0]['column_c0'], result[0]['column_c0_error'] = 0., 0.
            result[0]['column_c1'], result[0]['column_c1_error'] = 1., 0.
            result[0]['row_c0'], result[0]['row_c0_error'] = 0., 0.
            result[0]['row_c1'], result[0]['row_c1_error'] = 1., 0.
            result[0]['z'] = z_positions[0]
            for node in in_file_h5.root:
                indices = re.findall(r'\d+', node.name)
                dut_idx = int(indices[0])
                ref_idx = int(indices[1])
                result[dut_idx]['DUT'] = dut_idx
                dut_name = dut_names[dut_idx] if dut_names else ("DUT " + str(dut_idx))
                ref_name = dut_names[ref_idx] if dut_names else ("DUT " + str(ref_idx))
                logging.info('Aligning with %s data', node.name)

                if "column" in node.name.lower():
                    pixel_length_dut, pixel_length_ref = pixel_size[dut_idx][0], pixel_size[ref_idx][0]
                else:
                    pixel_length_dut, pixel_length_ref = pixel_size[dut_idx][1], pixel_size[ref_idx][1]

                data = node[:]

                # Start values for fitting
                mu_start = np.argmax(data, axis=1)
                A_start = np.max(data, axis=1)
                A_mean = np.mean(data, axis=1)

                # Get mu start values for the Gauss fits
                n_entries = np.sum(data, axis=1)
                mu_mean = np.zeros_like(n_entries)
                mu_mean[n_entries > 0] = np.average(data, axis=1, weights=range(0, data.shape[1]))[n_entries > 0] * sum(range(0, data.shape[1])) / n_entries[n_entries > 0]

                def gauss2(x, *p):
                    A, mu, sigma = p
                    return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))

                def res(p, y, x):
                    a_mean, a, mean, peak, sigma1, sigma2 = p
                    y_fit = gauss2(x, *(a_mean, mean, sigma1)) + gauss2(x, *(a, peak, sigma2))
                    err = y - y_fit
                    return err

                # Fit result arrays have -1.0 for a bad fit
                mean_fitted = np.array([-1.0 for _ in range(data.shape[0])])  # Peak of the Gaussfit
                mean_error_fitted = np.array([-1.0 for _ in range(data.shape[0])])  # Error of the fit of the peak
                sigma_fitted = np.array([-1.0 for _ in range(data.shape[0])])  # Sigma of the Gaussfit
                chi2 = np.array([-1.0 for _ in range(data.shape[0])])  # Chi2 of the fit
                n_cluster = np.array([-1.0 for _ in range(data.shape[0])])  # Number of cluster per bin

                # Loop over all row/row or column/column slices and fit a double gaussian or gaussian + offset to the profile
                # Get values with highest correlation for alignment fit; do this with channel indices, later convert to um
                # Origin pixel cluster mean is 1. / 1., since cluster start from 1, 1 not 0, 0

                x_hist_fit = np.arange(1.0, data.shape[1] + 1.0)  # x bin positions
                ref_beam_center = np.argmax(np.sum(data, axis=1))  # Get the beam spot for plotting

                for index in np.arange(data.shape[0]):  # Loop over x dimension of correlation hitogram
                    fit = None
                    try:
                        p = [A_mean[index], A_start[index], mu_mean[index], mu_start[index], 500.0, 5.0]  # FIXME: hard coded starting values
                        plsq = leastsq(res, p, args=(data[index, :], x_hist_fit), full_output=True)
                        y_fit = gauss2(x_hist_fit, plsq[0][0], plsq[0][2], plsq[0][4]) + gauss2(x_hist_fit, plsq[0][1], plsq[0][3], plsq[0][5])
                        if plsq[1] is None:
                            raise RuntimeError
                        mean_fitted[index] = plsq[0][3]
                        mean_error_fitted[index] = np.sqrt(np.abs(np.diag(plsq[1])))[3]
                        sigma_fitted[index] = np.abs(plsq[0][5])
                        n_cluster[index] = data[index, :].sum()
                        fit_type = 1
                    except RuntimeError:
                        try:
                            p0 = [A_start[index], mu_start[index], 5.0, A_mean[index], 0.0]  # FIXME: hard coded start value
                            coeff, var_matrix = curve_fit(gauss_offset, x_hist_fit, data[index, :], p0=p0)
                            y_fit = gauss_offset(x_hist_fit, *coeff)
                            mean_fitted[index] = coeff[1]
                            mean_error_fitted[index] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
                            sigma_fitted[index] = np.abs(coeff[2])
                            n_cluster[index] = data[index, :].sum()
                            fit_type = 2
                        except RuntimeError:
                            pass
                    finally:  # Create plot in the center of the mean data
                        if index == int(ref_beam_center):
                            plot_utils.plot_correlation_fit(x=x_hist_fit,
                                                            y=data[index, :],
                                                            y_fit=y_fit,
                                                            fit_type=fit_type,
                                                            xlabel='%s %s' % ("Column" if "column" in node.name.lower() else "Row", ref_name),
                                                            title="Correlation of %s: %s vs. %s at %s %d" % ("columns" if "column" in node.name.lower() else "rows",
                                                                                                             ref_name, dut_name, "column" if "column" in node.name.lower() else "row", index),
                                                            output_pdf=output_pdf)

                # Unset invalid data
                mean_fitted[~np.isfinite(mean_fitted)] = -1
                mean_error_fitted[~np.isfinite(mean_error_fitted)] = -1

                # Convert fit results to um for alignment fit
                mean_fitted *= pixel_length_ref
                mean_error_fitted = pixel_length_ref * mean_error_fitted

                # Show the straigt line correlation fit including fit errors and offsets from the fit
                # Let the user change the cuts (error limit, offset limit) and refit until result looks good
                refit = True
                selected_data = np.ones_like(mean_fitted, dtype=np.bool)
                x = np.arange(1.0, mean_fitted.shape[0] + 1.0) * pixel_length_dut
                actual_iteration = 0  # Refit counter for non interactive mode
                while(refit):
                    selected_data, fit, refit = plot_utils.plot_alignments(x=x,
                                                                           mean_fitted=mean_fitted,
                                                                           mean_error_fitted=mean_error_fitted,
                                                                           n_cluster=n_cluster,
                                                                           ref_name=ref_name,
                                                                           dut_name=dut_name,
                                                                           title="Correlation of %s: %s vs. %s" % ("columns" if "column" in node.name.lower() else "rows", dut_name, ref_name),
                                                                           non_interactive=non_interactive)
                    x = x[selected_data]
                    mean_fitted = mean_fitted[selected_data]
                    mean_error_fitted = mean_error_fitted[selected_data]
                    sigma_fitted = sigma_fitted[selected_data]
                    chi2 = chi2[selected_data]
                    n_cluster = n_cluster[selected_data]
                    # Stop in non interactive mode if the number of refits (iterations) is reached
                    if non_interactive:
                        actual_iteration += 1
                        if actual_iteration > iterations:
                            break

                # Linear fit, usually describes correlation very well, slope is close to 1.
                # With low energy beam and / or beam with diverse agular distribution, the correlation will not be perfectly straight

                praefix = 'column' if 'column' in node.name.lower() else 'row'

                if fix_slope:
                    def line(x, c0):
                        return c0 + 1. * x

                    # Use results from straight line fit as start values for this final fit
                    re_fit, re_fit_pcov = curve_fit(line, x, mean_fitted, sigma=mean_error_fitted, absolute_sigma=True, p0=[fit[0]])

                    # Write fit results to array
                    result[dut_idx]['%s_c0' % praefix], result[dut_idx]['%s_c0_error' % praefix] = re_fit[0], np.absolute(re_fit_pcov[0][0]) ** 0.5
                    result[dut_idx]['%s_c1' % praefix], result[dut_idx]['%s_c1_error' % praefix] = 1., 0.
                    result[dut_idx]['z'] = z_positions[dut_idx]
                else:
                    def line(x, c0, c1):
                        return c0 + c1 * x

                    # Use results from straight line fit as start values for this final fit
                    re_fit, re_fit_pcov = curve_fit(line, x, mean_fitted, sigma=mean_error_fitted, absolute_sigma=True, p0=[fit[0], fit[1]])

                    # Write fit results to array
                    result[dut_idx]['%s_c0' % praefix], result[dut_idx]['%s_c0_error' % praefix] = re_fit[0], np.absolute(re_fit_pcov[0][0]) ** 0.5
                    result[dut_idx]['%s_c1' % praefix], result[dut_idx]['%s_c1_error' % praefix] = re_fit[1], np.absolute(re_fit_pcov[1][1]) ** 0.5
                    result[dut_idx]['z'] = z_positions[dut_idx]

                # Calculate mean sigma (is a residual when assuming straight tracks) and its error and store the actual data in result array
                # This error is needed for track finding and track quality determination
                mean_sigma = pixel_length_ref * np.mean(np.array(sigma_fitted))
                mean_sigma_error = pixel_length_ref * np.std(np.array(sigma_fitted)) / np.sqrt(np.array(sigma_fitted).shape[0])

                result[dut_idx]['%s_sigma' % praefix], result[dut_idx]['%s_sigma_error' % praefix] = mean_sigma, mean_sigma_error

                # Plot selected data with fit
                fit_fn = np.poly1d(re_fit[::-1])
                plot_utils.plot_alignment_fit(x=x, mean_fitted=mean_fitted, fit_fn=fit_fn, fit=re_fit, pcov=re_fit_pcov, chi2=chi2, mean_error_fitted=mean_error_fitted, dut_name=dut_name, ref_name=ref_name, title="Correlation of %s: %s vs. %s" % ("columns" if "column" in node.name.lower() else "rows", ref_name, dut_name), output_pdf=output_pdf)

            logging.info('Store pre alignment data in %s', output_alignment_file)
            with tb.open_file(output_alignment_file, mode="w") as out_file_h5:
                try:
                    result_table = out_file_h5.create_table(out_file_h5.root, name='PreAlignment', description=result.dtype, title='Prealignment alignment from correlation', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    result_table.append(result)
                except tb.exceptions.NodeError:
                    logging.warning('Coarse alignment table exists already. Do not create new.')


def merge_cluster_data(input_cluster_files, output_merged_file, pixel_size, chunk_size=4999999):
    '''Takes the cluster from all cluster files and merges them into one big table aligned at a common event number.
    Empty entries are signaled with column = row = charge = 0.

    Alignment information from the alignment file is used to correct the column/row positions. If alignment is
    available it is used (translation/rotations for each plane), otherwise prealignment data (offset, slope of correlation)
    is used.

    Parameters
    ----------
    input_cluster_files : list of pytables files
        File name of the input cluster files with correlation data.
    input_alignment_file : pytables file
        File name of the input aligment data.
    output_merged_file : pytables file
        File name of the output tracklet file.
    limit_events : int
        Limit events to givien number. Only events with hits are counted. If None or 0, all events will be taken.
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
    description.extend([('track_quality', np.uint32), ('n_tracks', np.uint8)])

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
                merged_cluster_array = np.zeros((common_event_numbers.shape[0],), dtype=description)  # Result array to be filled. For no hit: column = row = 0

                # Set the event number
                merged_cluster_array['event_number'] = common_event_numbers[:]

                # Fill result array with DUT 0 data
                actual_cluster = analysis_utils.map_cluster(common_event_numbers, cluster_dut_0)
                selection = actual_cluster['mean_column'] != 0  # Add only real hits, 0 is a virtual hit

                merged_cluster_array['x_dut_0'][selection] = pixel_size[0][0] * actual_cluster['mean_column'][selection]  # Convert channel indices to um
                merged_cluster_array['y_dut_0'][selection] = pixel_size[0][1] * actual_cluster['mean_row'][selection]  # Convert channel indices to um
                merged_cluster_array['z_dut_0'][selection] = 0.
                merged_cluster_array['charge_dut_0'][selection] = actual_cluster['charge'][selection]

                # Fill result array with other DUT data
                # Second loop: get the cluster from all files and merge them to the common event number
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_cluster, start_indices_2[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start=start_indices_2[dut_index], start_event_number=actual_start_event_number, stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size):  # Loop over the cluster in the actual cluster file in chunks
                            actual_cluster = analysis_utils.map_cluster(common_event_numbers, actual_cluster)
                            selection = actual_cluster['mean_column'] != 0  # Add only real hits, 0 is a virtual hit
                            actual_mean_column = pixel_size[dut_index][0] * actual_cluster['mean_column'][selection]  # Convert channel indices to um
                            actual_mean_row = pixel_size[dut_index][1] * actual_cluster['mean_row'][selection]  # Convert channel indices to um

                            merged_cluster_array['x_dut_%d' % (dut_index)][selection] = actual_mean_column
                            merged_cluster_array['y_dut_%d' % (dut_index)][selection] = actual_mean_row
                            merged_cluster_array['z_dut_%d' % (dut_index)][selection] = 0.

                            merged_cluster_array['charge_dut_%d' % (dut_index)][selection] = actual_cluster['charge'][selection]

                np.nan_to_num(merged_cluster_array)
                merged_cluster_table.append(merged_cluster_array)
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(index)
            progress_bar.finish()


def apply_alignment(input_hit_file, input_alignment, output_hit_aligned_file, inverse=False, force_prealignment=False, reset_z=False, no_z=False, use_duts=None, chunk_size=1000000):
    ''' Takes a file with tables containing hit information (x, y, z) and applies the alignment to each DUT. The alignment data is used. If this is not
    available a fallback to the prealignment is done.

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
        Take the prealignment, although if a coarse alignment is availale
    reset_z : boolean
        Set z = 0 before applying alignment
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
                    logging.info('Use prealignment data from file')
    except TypeError:  # The input_alignment is an array
        alignment = input_alignment
        try:  # Check if array is prealignent array
            alignment['column_c0']
            logging.info('Use prealignment data')
            use_prealignment = True
        except ValueError:
            logging.info('Use alignment data')
            use_prealignment = False

    n_duts = alignment.shape[0]

    with tb.open_file(input_hit_file, mode='r') as in_file_h5:
        with tb.open_file(output_hit_aligned_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:
                hits = node
                new_node_name = hits.name

                if new_node_name == 'MergedCluster':  # Merged cluster with alignment are tracklets
                    new_node_name = 'Tracklets'

                hits_aligned_table = out_file_h5.create_table(out_file_h5.root, name=new_node_name, description=np.zeros((1,), dtype=hits.dtype).dtype, title=hits.title, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                for hits_chunk, _ in analysis_utils.data_aligned_at_events(hits, chunk_size=chunk_size):
                    for dut_index in range(0, n_duts):
                        if use_duts is not None and dut_index not in use_duts:  # omit DUT
                            continue
                        selection = hits_chunk['x_dut_%d' % dut_index] != 0  # Do not change vitual hits

                        if reset_z:
                            hits_chunk['z_dut_%d' % dut_index][selection] = 0

                        if use_prealignment:
                            c0_column = alignment[dut_index]['column_c0']
                            c1_column = alignment[dut_index]['column_c1']
                            c0_row = alignment[dut_index]['row_c0']
                            c1_row = alignment[dut_index]['row_c1']
                            z = alignment[dut_index]['z']

                            if inverse:
                                hits_chunk['x_dut_%d' % dut_index][selection] = (hits_chunk['x_dut_%d' % dut_index][selection] - c0_column) / c1_column
                                hits_chunk['y_dut_%d' % dut_index][selection] = (hits_chunk['y_dut_%d' % dut_index][selection] - c0_row) / c1_row
                                if not no_z:
                                    hits_chunk['z_dut_%d' % dut_index][selection] -= z
                            else:
                                hits_chunk['x_dut_%d' % dut_index][selection] = (c1_column * hits_chunk['x_dut_%d' % dut_index][selection] + c0_column)
                                hits_chunk['y_dut_%d' % dut_index][selection] = (c1_row * hits_chunk['y_dut_%d' % dut_index][selection] + c0_row)
                                if not no_z:
                                    hits_chunk['z_dut_%d' % dut_index][selection] += z
                        else:  # Apply transformation from fine alignment information
                            if inverse:
                                transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=alignment[dut_index]['translation_x'],
                                                                                                             y=alignment[dut_index]['translation_y'],
                                                                                                             z=alignment[dut_index]['translation_z'],
                                                                                                             alpha=alignment[dut_index]['alpha'],
                                                                                                             beta=alignment[dut_index]['beta'],
                                                                                                             gamma=alignment[dut_index]['gamma'])
                            else:
                                transformation_matrix = geometry_utils.local_to_global_transformation_matrix(x=alignment[dut_index]['translation_x'],
                                                                                                             y=alignment[dut_index]['translation_y'],
                                                                                                             z=alignment[dut_index]['translation_z'],
                                                                                                             alpha=alignment[dut_index]['alpha'],
                                                                                                             beta=alignment[dut_index]['beta'],
                                                                                                             gamma=alignment[dut_index]['gamma'])

                            hits_chunk['x_dut_%d' % dut_index][selection], hits_chunk['y_dut_%d' % dut_index][selection], z = geometry_utils.apply_transformation_matrix(x=hits_chunk['x_dut_%d' % dut_index][selection],
                                                                                                                                                                         y=hits_chunk['y_dut_%d' % dut_index][selection],
                                                                                                                                                                         z=hits_chunk['z_dut_%d' % dut_index][selection],
                                                                                                                                                                         transformation_matrix=transformation_matrix)
                            if not no_z:
                                hits_chunk['z_dut_%d' % dut_index][selection] = z

                    hits_aligned_table.append(hits_chunk)


def _fit_tracks_loop(track_hits):
    ''' Do 3d line fit and calculate chi2 for each fit. '''
    def line_fit_3d(hits):
        datamean = hits.mean(axis=0)
        offset, slope = datamean, np.linalg.svd(hits - datamean)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
        intersections = offset + slope / slope[2] * (hits.T[2][:, np.newaxis] - offset[2])  # Fitted line and DUT plane intersections (here: points)
        chi2 = np.sum(np.square(hits - intersections), dtype=np.uint32)  # Chi2 of the fit in um
        return datamean, slope, chi2

    slope = np.zeros((track_hits.shape[0], 3,))
    offset = np.zeros((track_hits.shape[0], 3,))
    chi2 = np.zeros((track_hits.shape[0],))

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        try:
            offset[index], slope[index], chi2[index] = line_fit_3d(actual_hits)
        except np.linalg.linalg.LinAlgError:
            chi2[index] = 1e9

    return offset, slope, chi2

# FIMXE: ALL FUNCTIONS BELOW NOT WORKING RIGHT NOW


def check_hit_alignment(input_tracklets_file, output_pdf_file, combine_n_hits=100000, correlated_only=False):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).

    Parameters
    ----------
    input_tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf_file : pdf file name object
    combine_n_hits : int
        The number of events to combine for the hit position check
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('=== Check hit alignment ===')
    with tb.open_file(input_tracklets_file, mode="r") as in_file_h5:
        with PdfPages(output_pdf_file) as output_fig:
            for table_column in in_file_h5.root.Tracklets.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    median, mean, std, alignment, correlation = [], [], [], [], []
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Check alignment for % s', table_column)
                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Tracklets.shape[0], term_width=80)
                    progress_bar.start()
                    for index in range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits):
                        particles = in_file_h5.root.Tracklets[index:index + combine_n_hits]
                        particles = particles[np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)]  # only select events with hits in both DUTs
                        if correlated_only is True:
                            particles = particles[particles['track_quality'] & (1 << (24 + int(table_column[-1]))) == (1 << (24 + int(table_column[-1])))]
                        if particles.shape[0] == 0:
                            logging.warning('No correlation for dut %s and tracks %d - %d', table_column, index, index + combine_n_hits)
                            median.append(-1)
                            mean.append(-1)
                            std.append(-1)
                            alignment.append(0)
                            correlation.append(0)
                            continue
                        difference = particles[:][ref_dut_column] - particles[:][table_column]

                        # Calculate median, mean and RMS
                        actual_median, actual_mean, actual_rms = np.median(difference), np.mean(difference), np.std(difference)
                        alignment.append(np.median(np.abs(difference)))
                        correlation.append(difference.shape[0] * 100. / combine_n_hits)

                        median.append(actual_median)
                        mean.append(actual_mean)
                        std.append(actual_rms)

                        plot_utils.plot_hit_alignment('Aligned position difference for events %d - %d' % (index, index + combine_n_hits), difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_fig, bins=64)
                        progress_bar.update(index)
                    plot_utils.plot_hit_alignment_2(in_file_h5, combine_n_hits, median, mean, correlation, alignment, output_fig)
                    progress_bar.finish()


def fix_event_alignment(input_tracklets_file, tracklets_corr_file, input_alignment_file, error=3., n_bad_events=100, n_good_events=10, correlation_search_range=20000, good_events_search_range=100):
    '''Description

    Parameters
    ----------
    input_tracklets_file: pytables file
        Input file with original Tracklet data
    tracklets_corr_file: pyables_file
        Output file for corrected Tracklet data
    input_alignment_file: pytables file
        File with alignment data (used to get alignment fit errors)
    error: float
        Defines how much deviation between reference and observed DUT hit is allowed
    n_bad_events: int
        Detect no correlation when n_bad_events straight are not correlated
    n_good_events: int
    good_events_search_range: int
        n_good_events out of good_events_search_range must be correlated to detect correlation
    correlation_search_range: int
        Number of events that get checked for correlation when no correlation is found
    '''

    # Get alignment errors
    with tb.open_file(input_alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        n_duts = int(correlations.shape[0] / 2 + 1)
        column_sigma = np.zeros(shape=n_duts)
        row_sigma = np.zeros(shape=n_duts)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, n_duts):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    logging.info('=== Fix event alignment ===')

    with tb.open_file(input_tracklets_file, mode="r") as in_file_h5:
        particles = in_file_h5.root.Tracklets[:]
        event_numbers = np.ascontiguousarray(particles['event_number'])
        ref_column = np.ascontiguousarray(particles['column_dut_0'])
        ref_row = np.ascontiguousarray(particles['row_dut_0'])
        ref_charge = np.ascontiguousarray(particles['charge_dut_0'])

        particles_corrected = np.zeros_like(particles)

        particles_corrected['track_quality'] = (1 << 24)  # DUT0 is always correlated with itself

        for table_column in in_file_h5.root.Tracklets.dtype.names:
            if 'column_dut' in table_column and 'dut_0' not in table_column:
                column = np.ascontiguousarray(particles[table_column])  # create arrays for event alignment fixing
                row = np.ascontiguousarray(particles['row_dut_' + table_column[-1]])
                charge = np.ascontiguousarray(particles['charge_dut_' + table_column[-1]])

                logging.info('Fix alignment for % s', table_column)
                correlated, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=error, n_bad_events=n_bad_events, n_good_events=n_good_events, correlation_search_range=correlation_search_range, good_events_search_range=good_events_search_range)
                logging.info('Corrected %d places in the data', n_fixes)
                particles_corrected['event_number'] = event_numbers  # create new particles array with corrected values
                particles_corrected['column_dut_0'] = ref_column  # copy values that have not been changed
                particles_corrected['row_dut_0'] = ref_row
                particles_corrected['charge_dut_0'] = ref_charge
                particles_corrected['n_tracks'] = particles['n_tracks']
                particles_corrected[table_column] = column  # fill array with corrected values
                particles_corrected['row_dut_' + table_column[-1]] = row
                particles_corrected['charge_dut_' + table_column[-1]] = charge

                correlation_index = np.where(correlated == 1)[0]

                # Set correlation flag in track_quality field
                particles_corrected['track_quality'][correlation_index] |= (1 << (24 + int(table_column[-1])))

        # Create output file
        with tb.open_file(tracklets_corr_file, mode="w") as out_file_h5:
            try:
                out_file_h5.root.Tracklets._f_remove(recursive=True, force=False)
                logging.warning('Overwrite old corrected Tracklets file')
            except tb.NodeError:
                logging.info('Create new corrected Tracklets file')

            correction_out = out_file_h5.create_table(out_file_h5.root, name='Tracklets', description=in_file_h5.root.Tracklets.description, title='Corrected Tracklets data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            correction_out.append(particles_corrected)


def align_z(input_track_candidates_file, input_alignment_file, output_pdf, z_positions=None, track_quality=1, max_tracks=3, warn_at=0.5):
    '''Minimizes the squared distance between track hit and measured hit by changing the z position.
    In a perfect measurement the function should be minimal at the real DUT position. The tracks is given
    by the first and last reference hit. A track quality cut is applied to all cuts first.

    Parameters
    ----------
    input_track_candidates_file : pytables file
    input_alignment_file : pytables file
    output_pdf : pdf file name object
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
    '''
    logging.info('=== Find relative z-position ===')

    def pos_error(z, dut, first_reference, last_reference):
        return np.mean(np.square(z * (last_reference - first_reference) + first_reference - dut))

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
            track_candidates = in_file_h5.root.TrackCandidates[::10]  # take only every 10th track

            results = np.zeros((n_duts - 2,), dtype=[('DUT', np.uint8), ('z_position_column', np.float32), ('z_position_row', np.float32)])

            for dut_index in range(1, n_duts - 1):
                logging.info('Find best z-position for DUT %d', dut_index)
                dut_selection = (1 << (n_duts - 1)) | 1 | ((1 << (n_duts - 1)) >> dut_index)
                good_track_selection = np.logical_and((track_candidates['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8)), track_candidates['n_tracks'] <= max_tracks)
                good_track_candidates = track_candidates[good_track_selection]

                first_reference_row, last_reference_row = good_track_candidates['row_dut_0'], good_track_candidates['row_dut_%d' % (n_duts - 1)]
                first_reference_col, last_reference_col = good_track_candidates['column_dut_0'], good_track_candidates['column_dut_%d' % (n_duts - 1)]

                z = np.arange(0, 1., 0.01)
                dut_row = good_track_candidates['row_dut_%d' % dut_index]
                dut_col = good_track_candidates['column_dut_%d' % dut_index]
                dut_z_col = minimize_scalar(pos_error, args=(dut_col, first_reference_col, last_reference_col), bounds=(0., 1.), method='bounded')
                dut_z_row = minimize_scalar(pos_error, args=(dut_row, first_reference_row, last_reference_row), bounds=(0., 1.), method='bounded')
                dut_z_col_pos_errors, dut_z_row_pos_errors = [pos_error(i, dut_col, first_reference_col, last_reference_col) for i in z], [pos_error(i, dut_row, first_reference_row, last_reference_row) for i in z]
                results[dut_index - 1]['DUT'] = dut_index
                results[dut_index - 1]['z_position_column'] = dut_z_col.x
                results[dut_index - 1]['z_position_row'] = dut_z_row.x

                plot_utils.plot_z(z, dut_z_col, dut_z_row, dut_z_col_pos_errors, dut_z_row_pos_errors, dut_index, output_fig)

    with tb.open_file(input_alignment_file, mode='r+') as out_file_h5:
        try:
            z_table_out = out_file_h5.createTable(out_file_h5.root, name='Zposition', description=results.dtype, title='Relative z positions of the DUTs without references', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            z_table_out.append(results)
        except tb.NodeError:
            logging.warning('Z position are do already exist. Do not overwrite.')

    z_positions_rec = np.add(([0.0] + results[:]['z_position_row'].tolist() + [1.0]), ([0.0] + results[:]['z_position_column'].tolist() + [1.0])) / 2.0

    if z_positions is not None:  # check reconstructed z against measured z
        z_positions_rec_abs = [i * z_positions[-1] for i in z_positions_rec]
        z_differences = [abs(i - j) for i, j in zip(z_positions, z_positions_rec_abs)]
        failing_duts = [j for (i, j) in zip(z_differences, range(5)) if i >= warn_at]
        logging.info('Absolute reconstructed z-positions %s', str(z_positions_rec_abs))
        if failing_duts:
            logging.warning('The reconstructed z positions are more than %1.1f cm off for DUTS %s', warn_at, str(failing_duts))
        else:
            logging.info('Difference between measured and reconstructed z-positions %s', str(z_differences))

    return z_positions_rec_abs if z_positions is not None else z_positions_rec


def alignment(input_track_candidates_file, input_alignment_file, n_pixels, pixel_size, ignore_duts=None, max_iterations=10, chunk_size=1000000):
    ''' This function does an alignment of the DUTs and sets translation and rotation values for all DUTs.
    The reference DUT defines the global coordinate system position at 0, 0, 0 and should be well in the beam and not heavily rotated.

    To solve the chicken-and-egg problem that a good dut alignment needs hits belonging to one track, but good track finding needs a good dut alignment this
    function work only on already prealigned hits belonging to one track. Thus this function can be called only after track finding.

    These steps are done
    1. Take the found tracks and revert partially the prealignment
    2. Take the track hits belonging to one track and fit tracks for all DUTs
    3. Calculate the residuals for each DUT
    4. Deduce rotations from the residuals and apply them to the hits
    5. Deduce the translation of each plane
    6. Store the alignment information

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
    ignore_duts : iterable
        the duts that are not taken in a fit. Needed to exclude small planes / very bad working planes from the fit
        to make a good alignment possible
    max_iterations : number
        Maximum iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations)
    chunk_size: int
        Defines the amount of in-RAM data. The higher the more RAM is used and the faster this function works.
    '''

    logging.info('=== Aligning DUTs ===')

    def get_rotation_from_residual_fit(m_xx, m_xy, m_yx, m_yy):
        tangamma = m_xy / (1 - np.abs(m_xx))
        singamma = np.sign(tangamma) * np.sqrt(tangamma ** 2 / (1 + tangamma ** 2))
        gamma = np.arcsin(singamma)
        cosbeta = (1 - np.abs(m_xx)) / np.sqrt(1 - singamma ** 2)
        if cosbeta > 1:
            cosbeta = 1 - (cosbeta - 1)  # TODO: check if this is ok
        beta = np.arcsin(np.sqrt(1 - cosbeta ** 2))
        cosalpha = (-np.abs(m_yy) - tangamma * m_yx + 1) / (np.sqrt(1 - singamma ** 2) + singamma * tangamma)
        if cosalpha > 1:
            cosalpha = 1 - (cosalpha - 1)  # TODO: check if this is ok
        alpha = np.arcsin(np.sqrt(1 - cosalpha ** 2))

        return alpha, beta, gamma

    def reconstruct_translation(column_0, row_0, column_1, row_1):
        ''' Calculated the median of the column / row difference.
        These values are the translation in x/y.
        '''
        return np.median(column_0 - column_1), np.median(row_0 - row_1)

    def get_percentile_for_distribution(data, percentile=68.2):
        ''' FIXME: Overly complex function to return the +- range from the mean of the data where the defined
        percentile amount of values are within.
        E.g.: normal distribution with sigma = 1 with percentile 68.2 returns 1.
        '''
        median, std = np.median(data), np.std(data)
        hist, bin_edges = np.histogram(data, bins=1000, range=(median - 5. * std, median + 5. * std))

        bin_width = bin_edges[1] - bin_edges[0]
        mean_index = np.where(np.logical_and(bin_edges >= median - bin_width, bin_edges <= median + bin_width))[0]

        if mean_index.shape[0] > 1:
            mean_index = mean_index[0]

        sum_total = np.sum(hist)
        for i in range(hist.shape[0]):
            actual_sum = hist[mean_index - i:mean_index + i].sum()
            if actual_sum > sum_total * percentile / 100.:
                break

        return bin_edges[mean_index + i + 1] - median

    def analyze_residuals(residuals_file_h5, output_fig, n_duts, plot_title_praefix=''):
        logging.info('Alignment step 4: Deduce rotations from the residuals')
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
        alignment_parameters = np.zeros((n_duts,), dtype=description)

        total_residual = 0  # Sum of ll residuals to judge the overall alignment

        with tb.open_file(residuals_file_h5) as in_file_h5:
            for dut_index in range(n_duts):
                # Global residuals
                hist_node = in_file_h5.get_node('/ResidualsX_DUT%d' % dut_index)
                hist_residual_x = (hist_node[:], np.linspace(hist_node._v_attrs.x_edges[0], hist_node._v_attrs.x_edges[-1], num=hist_node[:].shape[0] + 1))
                coeff, var_matrix = None, None
                try:
                    coeff, var_matrix = curve_fit(analysis_utils.gauss, hist_residual_x[1][:-1], hist_residual_x[0], p0=[np.amax(hist_residual_x[0]), 0, 100])
                except:  # Fit error
                    pass
                total_residual = np.sqrt(np.square(total_residual) + np.square(coeff[2]))

                if output_fig:
                    plot_utils.plot_residuals(histogram=hist_residual_x,
                                              fit=coeff,
                                              fit_errors=var_matrix,
                                              title='Residuals for DUT %d' % dut_index,
                                              x_label='X residual [um]',
                                              output_fig=output_fig)

                hist_node = in_file_h5.get_node('/ResidualsY_DUT%d' % dut_index)
                hist_residual_y = (hist_node[:], np.linspace(hist_node._v_attrs.x_edges[0], hist_node._v_attrs.x_edges[-1], num=hist_node[:].shape[0] + 1))
                coeff, var_matrix = None, None
                try:
                    coeff, var_matrix = curve_fit(analysis_utils.gauss, hist_residual_y[1][:-1], hist_residual_y[0], p0=[np.amax(hist_residual_y[0]), 0, 100])
                except:  # Fit error
                    pass
                total_residual = np.sqrt(np.square(total_residual) + np.square(coeff[2]))
                
                if output_fig:
                    plot_utils.plot_residuals(histogram=hist_residual_y,
                                              fit=coeff,
                                              fit_errors=var_matrix,
                                              title='Residuals for DUT %d' % dut_index,
                                              x_label='Y residual [um]',
                                              output_fig=output_fig)

                # Fit position residuals with a line
                line = lambda x, c0, c1: c0 + c1 * x

                def get_median(array, values):  # Calculate the median of a 2D histogram along axis=1 with given values
                    def find_nearest(array, value):
                        return (np.abs(array - value[:, np.newaxis])).argmin(axis=1)
                    cumsum = np.cumsum(array, axis=1)
                    idx = find_nearest(cumsum, np.max(cumsum, axis=1) / 2.)
                    return values[idx]

                hist_x_residual_x = in_file_h5.get_node('/XResidualsX_DUT%d' % dut_index)
                x_edges = hist_x_residual_x._v_attrs.x_edges
                y_edges = hist_x_residual_x._v_attrs.y_edges
                x = np.linspace(x_edges[0], x_edges[-1], num=hist_x_residual_x.shape[0])
                values = np.linspace(y_edges[0], y_edges[-1], num=hist_x_residual_x.shape[1])
                n_hits = hist_x_residual_x[:].sum(axis=1)
                y = get_median(hist_x_residual_x, values)
                n_hits_threshold = np.percentile(hist_x_residual_x[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
                x_fit = x[n_hits > n_hits_threshold]
                y_fit = y[n_hits > n_hits_threshold]
                popt, pcov = curve_fit(line, x_fit, y_fit)  # Fit straight line
                if output_fig:
                    plot_utils.plot_position_residuals((hist_x_residual_x, x_edges, y_edges), x, y, x_label='X position [um]', y_label='X residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
                m_xx = popt[1]

                hist_y_residual_y = in_file_h5.get_node('/YResidualsY_DUT%d' % dut_index)
                x_edges = hist_y_residual_y._v_attrs.x_edges
                y_edges = hist_y_residual_y._v_attrs.y_edges
                x = np.linspace(x_edges[0], x_edges[-1], num=hist_y_residual_y.shape[0])
                values = np.linspace(y_edges[0], y_edges[-1], num=hist_y_residual_y.shape[1])
                n_hits = hist_y_residual_y[:].sum(axis=1)
                y = get_median(hist_y_residual_y, values)
                n_hits_threshold = np.percentile(hist_y_residual_y[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
                x_fit = x[n_hits > n_hits_threshold]
                y_fit = y[n_hits > n_hits_threshold]
                popt, pcov = curve_fit(line, x_fit, y_fit)  # Fit straight line
                if output_fig:
                    plot_utils.plot_position_residuals((hist_y_residual_y, x_edges, y_edges), x, y, x_label='Y position [um]', y_label='Y residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
                m_yy = popt[1]

                hist_x_residual_y = in_file_h5.get_node('/XResidualsY_DUT%d' % dut_index)
                x_edges = hist_x_residual_y._v_attrs.x_edges
                y_edges = hist_x_residual_y._v_attrs.y_edges
                x = np.linspace(x_edges[0], x_edges[-1], num=hist_x_residual_y.shape[0])
                values = np.linspace(y_edges[0], y_edges[-1], num=hist_x_residual_y.shape[1])
                n_hits = hist_x_residual_y[:].sum(axis=1)
                y = get_median(hist_x_residual_y, values)
                n_hits_threshold = np.percentile(hist_x_residual_y[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
                x_fit = x[n_hits > n_hits_threshold]
                y_fit = y[n_hits > n_hits_threshold]
                popt, pcov = curve_fit(line, x_fit, y_fit)  # Fit straight line
                if output_fig:
                    plot_utils.plot_position_residuals((hist_x_residual_y, x_edges, y_edges), x, y, x_label='X position [um]', y_label='Y residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
                m_xy = popt[1]

                hist_y_residual_x = in_file_h5.get_node('/YResidualsX_DUT%d' % dut_index)
                x_edges = hist_y_residual_x._v_attrs.x_edges
                y_edges = hist_y_residual_x._v_attrs.y_edges
                x = np.linspace(x_edges[0], x_edges[-1], num=hist_y_residual_x.shape[0])
                values = np.linspace(y_edges[0], y_edges[-1], num=hist_y_residual_x.shape[1])
                n_hits = hist_y_residual_x[:].sum(axis=1)
                y = get_median(hist_y_residual_x, values)
                n_hits_threshold = np.percentile(hist_y_residual_x[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
                x_fit = x[n_hits > n_hits_threshold]
                y_fit = y[n_hits > n_hits_threshold]
                popt, pcov = curve_fit(line, x_fit, y_fit)  # Fit straight line
                if output_fig:
                    plot_utils.plot_position_residuals((hist_y_residual_x, x_edges, y_edges), x, y, x_label='Y position [um]', y_label='X residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
                m_yx = popt[1]

                alpha, beta, gamma = get_rotation_from_residual_fit(m_xx, m_xy, m_yx, m_yy)

                alignment_parameters[dut_index]['alpha'] = alpha
                alignment_parameters[dut_index]['beta'] = beta
                alignment_parameters[dut_index]['gamma'] = gamma

        return alignment_parameters, total_residual

    def store_alignment_parameters(alignment_file, alignment_parameters, mode='absolute'):
        if 'absolute' not in mode and 'relative' not in mode:
            raise RuntimeError('Mode %s is unknown', str(mode))
        with tb.open_file(alignment_file, mode="r+") as out_file_h5:  # Open file with alignment data
            alignment_parameters[:]['translation_z'] = out_file_h5.root.PreAlignment[:]['z']  # Set z from prealignment
            try:
                alignment_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', title='Table containing the alignment geometry parameters (translations and rotations)', description=np.zeros((1,), dtype=alignment_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                alignment_table.append(alignment_parameters)
            except tb.NodeError:
                if mode == 'absolute':
                    logging.warning('Overwrite existing alignment!')
                    out_file_h5.root.Alignment._f_remove()  # Remove old node, is there a better way?
                    alignment_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', title='Table containing the alignment geometry parameters (translations and rotations)', description=np.zeros((1,), dtype=alignment_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    alignment_table.append(alignment_parameters)
                else:
                    logging.info('Merge new alignment with old alignment')
                    old_alignment = out_file_h5.root.Alignment[:]
                    out_file_h5.root.Alignment._f_remove()  # Remove old node, is there a better way?
                    alignment_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', title='Table containing the alignment geometry parameters (translations and rotations)', description=np.zeros((1,), dtype=alignment_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    new_alignment = old_alignment
                    new_alignment['translation_x'] += alignment_parameters['translation_x']
                    new_alignment['translation_y'] += alignment_parameters['translation_y']
                    new_alignment['alpha'] += alignment_parameters['alpha']
                    new_alignment['beta'] += alignment_parameters['beta']
                    new_alignment['gamma'] += alignment_parameters['gamma']

                    new_alignment['alpha'] -= np.mean(new_alignment['alpha'])
                    new_alignment['beta'] -= np.mean(new_alignment['beta'])
                    new_alignment['gamma'] -= np.mean(new_alignment['gamma'])
                    alignment_table.append(new_alignment)

    # Step 1: Take the found tracks and revert the slope of the prealignment but keep the offset to ease track fitting
    logging.info('= Alignment step 1: Revert prealignment =')

    # Open the prealignment and reset offset
    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        alignment = in_file_h5.root.PreAlignment[:]
        # Reset offset to not revert the offset
        alignment['column_c0'] = 0.
        alignment['row_c0'] = 0.
        alignment['z'] = 0.
        n_duts = alignment.shape[0]

    # Revert prealignent (= revert the slope, offsets ore kept since they do not interfere with rotation estimation)
    apply_alignment(input_hit_file=input_track_candidates_file,
                    input_alignment=alignment,
                    output_hit_aligned_file=input_track_candidates_file[:-3] + '_no_align_tmp_0.h5',
                    inverse=True,
                    force_prealignment=True,
                    chunk_size=chunk_size)

    # Step 2: Fit tracks for all DUTs
    logging.info('= Alignment step 2 / iteration 0: Fit tracks for all DUTs =')
    fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_no_align_tmp_0.h5',
               input_alignment_file=input_alignment_file,
               output_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_0.h5',
               track_quality=1,
               include_duts=None,
               ignore_duts=ignore_duts,
               force_prealignment=True)

    # Step 3: Calculate the residuals for each DUT
    logging.info('= Alignment step 3 / iteration 0: Calculate the residuals for each DUT =')
    calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_0.h5',
                        input_alignment_file=input_alignment_file,
                        output_residuals_file=input_track_candidates_file[:-3] + '_residuals_0.h5',
                        n_pixels=n_pixels,
                        pixel_size=pixel_size,
                        force_prealignment=True,
                        output_pdf=False,
                        chunk_size=chunk_size)

    # Step 4: Deduce rotations from the residuals
    logging.info('= Alignment step 4 / iteration 0: Deduce rotations from the residuals =')
    with PdfPages(input_track_candidates_file[:-3] + '_Residuals_Stage_0.pdf') as output_pdf:
        alignment_parameters, total_residual = analyze_residuals(input_track_candidates_file[:-3] + '_residuals_0.h5',
                                                                 output_pdf,
                                                                 n_duts,
                                                                 plot_title_praefix='Before alignment')

    # Step 5: Set rotation information in alignment file
    logging.info('= Alignment step 5/ iteration 0: Set rotation information in alignment file =')
    store_alignment_parameters(input_alignment_file,
                               alignment_parameters,
                               mode='absolute')

    for iteration in range(1, 10):
        logging.info('= Alignment step 6 / iteration %d: Apply rotations =', iteration)
        apply_alignment(input_hit_file=input_track_candidates_file[:-3] + '_no_align_tmp_0.h5',  # Always apply alignment to starting file
                        input_alignment=input_alignment_file,
                        output_hit_aligned_file=input_track_candidates_file[:-3] + '_no_align_tmp_%d.h5' % iteration,
                        reset_z=True,
                        chunk_size=chunk_size)

        # Step 2: Fit tracks for all DUTs
        logging.info('= Alignment step 2 / iteration %d: Fit tracks for all DUTs =', iteration)
        fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_no_align_tmp_%d.h5' % iteration,
                   input_alignment_file=input_alignment_file,
                   output_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_%d.h5' % iteration,
                   include_duts=None,
                   ignore_duts=ignore_duts,
                   track_quality=1)

        # Step 3: Calculate the residuals for each DUT
        logging.info('= Alignment step 3 / iteration %d: Calculate constrained residuals for each DUT =', iteration)
        calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_%d.h5' % iteration,
                            input_alignment_file=input_alignment_file,
                            output_residuals_file=input_track_candidates_file[:-3] + '_residuals_%d.h5' % iteration,
                            n_pixels=n_pixels,
                            pixel_size=pixel_size,
                            output_pdf=False,
                            chunk_size=chunk_size)

        # Step 4: Deduce rotations from the residuals
        logging.info('= Alignment step 4 / iteration %d: Deduce rotations from the residuals =', iteration)
        alignment_parameters, new_total_residual = analyze_residuals(input_track_candidates_file[:-3] + '_residuals_%d.h5' % iteration,
                                                                     output_fig=None,  # Do not plot to save time
                                                                     n_duts=n_duts)

        if new_total_residual > total_residual:
            logging.info('Best alignment found')
            break

        if iteration >= max_iterations:
            raise RuntimeError('Did not converge to good solution in %d iterations. Increase max_iterations', iteration)

        total_residual = new_total_residual

        # Step 5: Set rotation information in alignment file
        logging.info('= Alignment step 5 / iteration %d: Set rotation information in alignment file =', iteration)
        store_alignment_parameters(input_alignment_file,
                                   alignment_parameters,
                                   mode='relative')

    # Calculate and plot final result
    logging.info('= Alignment step 6: Check rotations result with unconstrained residuals and plot result =')
    apply_alignment(input_hit_file=input_track_candidates_file[:-3] + '_no_align_tmp_0.h5',  # Always apply alignment to starting file
                    input_alignment=input_alignment_file,
                    output_hit_aligned_file=input_track_candidates_file[:-3] + '_no_align_tmp_final.h5',
                    reset_z=True,
                    chunk_size=chunk_size)
    logging.info('= Alignment step 2 / iteration final: Fit tracks for all DUTs =')
    fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_no_align_tmp_final.h5',
               input_alignment_file=input_alignment_file,
               output_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_final.h5',
               track_quality=1)
    calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_final.h5',
                        input_alignment_file=input_alignment_file,
                        output_residuals_file=input_track_candidates_file[:-3] + '_residuals_final.h5',
                        n_pixels=n_pixels,
                        pixel_size=pixel_size,
                        output_pdf=False,
                        chunk_size=chunk_size)
    with PdfPages(input_track_candidates_file[:-3] + '_Residuals_Final.pdf') as output_pdf:
        analyze_residuals(input_track_candidates_file[:-3] + '_residuals_final.h5',
                          output_pdf,
                          n_duts,
                          plot_title_praefix='Alignment final result')

#     with PdfPages(input_track_candidates_file[:-3] + '_Residuals_Stage_Final.pdf') as output_pdf:
#         for dut in range(6):
#             apply_alignment(input_hit_file=input_track_candidates_file[:-3] + '_no_align_tmp_0.h5',
#                             input_alignment=input_alignment_file,
#                             output_hit_aligned_file=input_track_candidates_file[:-3] + '_dut_%d.h5' % dut,
#                             inverse=False,
#                             reset_z=True,
#                             use_duts=[dut],
#                             chunk_size=chunk_size)
#
# Step 2: Fit tracks for all DUTs
#             logging.info('= Alignment step 2: Fit tracks for all DUTs =')
#             fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_dut_%d.h5' % dut,
#                        input_alignment_file=input_alignment_file,
#                        output_tracks_file=input_track_candidates_file[:-3] + '_tracks_dut_%d.h5' % dut,
#                        fit_duts=[dut],
#                        track_quality=1)
#
#             logging.info('= Alignment step 3: Calculate the residuals for each DUT =')
#             calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_dut_%d.h5' % dut,
#                                 input_alignment_file=input_alignment_file,
#                                 output_residuals_file=input_track_candidates_file[:-3] + '_residuals_dut_%d.h5' % dut,
#                                 n_pixels=n_pixels,
#                                 pixel_size=pixel_size,
#                                 output_pdf=False,
#                                 use_duts=[dut],
#                                 chunk_size=chunk_size)
#
#             analyze_residuals(input_track_candidates_file[:-3] + '_residuals_dut_%d.h5' % dut, output_pdf, n_duts, plot_title_praefix='After alignment')

#             apply_alignment(input_hit_file=input_track_candidates_file,
#                             input_alignment=input_alignment_file,
#                             output_hit_aligned_file=input_track_candidates_file[:-3] + '_no_align_tmp_2.h5',
#                             inverse=False,
#                             no_z=True,
#                             chunk_size=chunk_size)
#
# Step 2: Fit tracks for all DUTs
#     logging.info('Alignment step 2: Fit tracks for all DUTs')
#     fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_no_align_tmp_2.h5',
#                input_alignment_file=input_alignment_file,
#                output_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_2.h5',
#                force_prealignment=True,
#                track_quality=1)
#
#     logging.info('Alignment step 3: Calculate the residuals for each DUT')
#     calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_tmp_2.h5',
#                         input_alignment_file=input_alignment_file,
#                         output_residuals_file=input_track_candidates_file[:-3] + '_residuals_2.h5',
#                         n_pixels=n_pixels,
#                         pixel_size=pixel_size,
#                         output_pdf=True,
#                         force_prealignment=True,
#                         chunk_size=chunk_size)


# Step 3: Find tracks from the tracklets and store them with quality indicator into track candidates table
#     logging.info('Coarse alignment step 3: Find tracks')
#     track_analysis.find_tracks(input_tracklets_file=output_alignment_file[:-3] + '_Tracklets_tmp.h5',
#                                input_alignment_file=output_alignment_file,
#                                output_track_candidates_file=output_alignment_file[:-3] + '_TrackCandidates_tmp.h5')
#
#     with tb.open_file(output_alignment_file, mode='r+') as io_file_alignment_h5:
#         prealignment = io_file_alignment_h5.root.Prealignment[:]
#
# Step 5: Take the track cluster without prealignment and find the best rotation to the reference DUT. Use the prealignment for starting values and range cuts.
#         with tb.open_file(output_alignment_file[:-3] + '_TrackCandidates_tmp.h5', mode='r') as in_file_h5:
#             logging.info('Coarse alignment step 5: Select cluster for coarse alignment')
# track_cluster = []  # Track hits used for coarse correlation per DUT
#             for track_candidates_chunk, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates_noalign, chunk_size=chunk_size):
#                 for dut_index in range(1, n_duts):
# real_hits_selection = np.logical_and(track_candidates_chunk['x_dut_%d' % dut_index] != 0, track_candidates_chunk['x_dut_0'] != 0)  # Do not select vitual hits
#
# dut_selection = 1  # DUTs to be used in the fit, DUT 0 is always included
# dut_selection |= (1 << dut_index)  # Add actual DUT
# track_quality = 1  # Take hits from tracklets where hits are within 2 sigma of the correlation; TODO: check if this biases the alignment result
#                     good_tracklets_selection = (track_candidates_chunk['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8))
#
#                     selection = np.logical_and(real_hits_selection, good_tracklets_selection)
#
#                     actual_tracklets = np.column_stack((track_candidates_chunk['x_dut_0'][selection],
#                                                         track_candidates_chunk['y_dut_0'][selection],
#                                                         track_candidates_chunk['x_dut_%d' % dut_index][selection],
#                                                         track_candidates_chunk['y_dut_%d' % dut_index][selection]))
#
# Select ~ 100000 random hits per DUT only to increase speed; this selection seems to be a good trade off between speed and accuracy, TODO: check
# actual_tracklets_indices = np.arange(actual_tracklets.shape[0])  # Indices of hits
#                     actual_tracklets = actual_tracklets[actual_tracklets_indices][:100000 / n_chunks]
#
#                     try:
#                         track_cluster[dut_index - 1] = np.append(track_cluster[dut_index - 1], actual_tracklets, axis=0)
#                     except IndexError:
#                         track_cluster.append(actual_tracklets)
#
# Transilation / rotation table from coarse alignment
# Is overwritten later in the fine alignment step; initial values for the reference DUT define no translation and no rotation
#             description = [('DUT', np.int)]
#             description.append(('translation_x', np.float))
#             description.append(('translation_y', np.float))
#             description.append(('translation_z', np.float))
#             description.append(('alpha', np.float))
#             description.append(('beta', np.float))
#             description.append(('gamma', np.float))
#             description.append(('correlation_x', np.float))
#             description.append(('correlation_y', np.float))
#
#             alignment_parameters = np.zeros((n_duts,), dtype=description)
#
# FIXME: the reference rotations have to be changed too
#             logging.info('Reconstruct rotation angles')
# for dut_index, dut_track_cluster in enumerate(track_cluster, start=1):  # Loop over DUTs
#                 logging.info('Deduce rotation of DUT %s', dut_index)
#                 alignment_parameters[dut_index]['DUT'] = dut_index
#                 column_0, row_0, column_1, row_1 = dut_track_cluster[:, 0], dut_track_cluster[:, 1], dut_track_cluster[:, 2], dut_track_cluster[:, 3]
#
#                 if angles is None:
#                     alpha, beta, gamma = 0., 0., 0.
#                 else:
#                     alpha, beta, gamma = angles[dut_index]
#
# Get slope from prealignment to set alpha, beta starting angle
#                 slope = prealignment[prealignment['dut_x'] == dut_index]['c1']
#
#                 if slope[0] < 0. or slope[1] < 0.:
#                     raise RuntimeWarning('Inverted DUTs are not tested yet. Reconstruction might not work!')
#                 if slope[0] < 1.:
#                     beta = np.arccos(slope[0])
#                 else:
#                     beta = np.arccos(1. - slope[0])
#                 if slope[0] > 1.1:
#                     raise RuntimeWarning('The reference DUT seems to be tilted. This is not tested yet. Reconstruction might not work!')
#                 if slope[1] < 1.:
#                     alpha = np.arccos(slope[1])
#                 else:
#                     alpha = np.arccos(1. - slope[1])
#                 if slope[1] > 1.1:
#                     raise RuntimeWarning('The reference DUT seems to be tilted. This is not tested yet. Reconstruction might not work!')
#
#                 alpha, beta, gamma = reconstruct_rotation(column_0=column_0,
#                                                           row_0=row_0,
#                                                           column_1=column_1,
#                                                           row_1=row_1,
#                                                           x0=(alpha, beta, gamma),
#                                                           errors=errors)
#
# Step 6: Apply the rotations to the hits to deduce the translations
#                 actual_column_1, actual_row_1, _ = geometry_utils.apply_rotation_matrix(x=column_1,
#                                                                                         y=row_1,
#                                                                                         z=np.zeros_like(column_1),
#                                                                                         rotation_matrix=geometry_utils.rotation_matrix(alpha, beta, gamma))
# Reconstruct translation (position) of the plane
#                 translation_x, translation_y = reconstruct_translation(column_0, row_0, actual_column_1, actual_row_1)
#                 translation_z = z_positions[dut_index] - z_positions[0]
#
# Save results for actual DUT
#                 alignment_parameters[dut_index]['translation_x'] = translation_x
#                 alignment_parameters[dut_index]['translation_y'] = translation_y
#                 alignment_parameters[dut_index]['translation_z'] = translation_z
#                 alignment_parameters[dut_index]['alpha'] = alpha
#                 alignment_parameters[dut_index]['beta'] = beta
#                 alignment_parameters[dut_index]['gamma'] = gamma
#
# Step 7: Check the coarse alignment and calculate correlation sigma in x / y needed for track quality assignment
#         logging.info('Coarse alignment step 7:  Check the coarse alignment and calculate sigma correlation needed for track quality assignment')
#         if not output_pdf_file:
#             output_pdf_file = os.path.splitext(output_alignment_file)[0] + '.pdf'
#
#         with PdfPages(output_pdf_file) as output_pdf:
#             logging.info('Select cluster for coarse alignment check')
#             with tb.open_file(output_alignment_file[:-3] + '_TrackCandidates_tmp.h5', mode='r') as in_file_h5:
# track_cluster = []  # Track hits used to check coarse correlation per DUT
#                 for track_candidates_chunk, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates_noalign, chunk_size=chunk_size):
#                     for dut_index in range(1, n_duts):
# selection = np.logical_and(track_candidates_chunk['x_dut_%d' % dut_index] != 0, track_candidates_chunk['y_dut_0'] != 0)  # Do not select vitual hits
#
#                         actual_tracklets = np.column_stack((track_candidates_chunk['x_dut_0'][selection],
#                                                             track_candidates_chunk['y_dut_0'][selection],
#                                                             track_candidates_chunk['x_dut_%d' % dut_index][selection],
#                                                             track_candidates_chunk['y_dut_%d' % dut_index][selection]))
#
# Select ~ 100000 random hits per DUT only to increase speed; this selection seems to be a good trade off between speed and accuracy, TODO: check
# actual_tracklets_indices = np.arange(actual_tracklets.shape[0])  # Indices of hits
#                         actual_tracklets = actual_tracklets[actual_tracklets_indices][:100000 / n_chunks]
#
#                         try:
#                             track_cluster[dut_index - 1] = np.append(track_cluster[dut_index - 1], actual_tracklets, axis=0)
#                         except IndexError:
#                             track_cluster.append(actual_tracklets)
#
# for dut_index, dut_track_cluster in enumerate(track_cluster, start=1):  # Loop over DUTs
#                     logging.info('Check of DUT %s', dut_index)
#                     column_0, row_0, column_1, row_1 = dut_track_cluster[:, 0], dut_track_cluster[:, 1], dut_track_cluster[:, 2], dut_track_cluster[:, 3]
#
#                     transformation_matrix = geometry_utils.local_to_global_transformation_matrix(x=alignment_parameters[dut_index]['translation_x'],
#                                                                                                  y=alignment_parameters[dut_index]['translation_y'],
#                                                                                                  z=alignment_parameters[dut_index]['translation_z'],
#                                                                                                  alpha=alignment_parameters[dut_index]['alpha'],
#                                                                                                  beta=alignment_parameters[dut_index]['beta'],
#                                                                                                  gamma=alignment_parameters[dut_index]['gamma'])
#
#                     column_1, row_1, _ = geometry_utils.apply_transformation_matrix(x=column_1,
#                                                                                     y=row_1,
#                                                                                     z=np.zeros_like(column_1),
#                                                                                     transformation_matrix=transformation_matrix)
#
# Warn for large offsets (> 10% of pixel pitch)
#                     if np.median(column_1 - column_0) > 0.1 * pixel_size[dut_index][0]:
#                         logging.warning('The difference between the column position of the DUT %d and the reference is large: %1.2f um!', dut_index, np.median(column_1 - column_0))
#                     if np.median(row_1 - row_0) > 0.1 * pixel_size[dut_index][1]:
#                         logging.warning('The difference between the row position of the DUT %d and the reference is large: %1.2f um!', dut_index, np.median(row_1 - row_0))
#
#                     alignment_parameters[dut_index]['correlation_x'] = get_percentile_for_distribution(column_1 - column_0)
#                     alignment_parameters[dut_index]['correlation_y'] = get_percentile_for_distribution(row_1 - row_0)
#
#                     plot_utils.plot_coarse_alignment_check(column_0, column_1, row_0, row_1, alignment_parameters[dut_index]['correlation_x'], alignment_parameters[dut_index]['correlation_y'], dut_index, output_pdf)
#
# Step 8: Store the coarse alignment data (rotation / translation data for each DUT)
#         logging.info('Coarse alignment step 8: Store the coarse alginment data (rotation / translation data for each DUT)')
#         try:
#             geometry_table = io_file_alignment_h5.create_table(io_file_alignment_h5.root, name='Alignment', title='File containing the alignment geometry parameters (translations and rotations)', description=np.zeros((1,), dtype=alignment_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
#             geometry_table.append(alignment_parameters)
#         except tb.exceptions.NodeError:
#             logging.warning('Alignment data exists already. Do not create new.')
#
#     logging.info('Coarse alignment step 9: Store the coarse alginment data (rotation / translation data for each DUT)')
