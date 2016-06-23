''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import re
import os
import progressbar
import warnings

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
from testbeam_analysis import track_analysis

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

    with PdfPages(os.path.join(os.path.dirname(os.path.abspath(output_alignment_file)), 'Prealignment.pdf')) as output_pdf:
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


def apply_alignment(input_hit_file, input_alignment, output_hit_aligned_file, inverse=False, force_prealignment=False, no_z=False, use_duts=None, chunk_size=1000000):
    ''' Takes a file with tables containing hit information (x, y, z) and applies the alignment to each DUT hits positions. The alignment data is used. If this is not
    available a fallback to the prealignment is done.
    One can also inverse the alignment or apply the alignment without changing the z position.

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

                        if use_prealignment:
                            hits_chunk['x_dut_%d' % dut_index][selection], hits_chunk['y_dut_%d' % dut_index][selection], hit_z = geometry_utils.apply_alignment(hits_x=hits_chunk['x_dut_%d' % dut_index][selection],
                                                                                                                                                                 hits_y=hits_chunk['y_dut_%d' % dut_index][selection],
                                                                                                                                                                 hits_z=hits_chunk['z_dut_%d' % dut_index][selection],
                                                                                                                                                                 dut_index=dut_index,
                                                                                                                                                                 prealignment=alignment,
                                                                                                                                                                 inverse=inverse)
                            if not no_z:
                                hits_chunk['z_dut_%d' % dut_index][selection] = hit_z
                        else:  # Apply transformation from fine alignment information
                            hits_chunk['x_dut_%d' % dut_index][selection], hits_chunk['y_dut_%d' % dut_index][selection], hit_z = geometry_utils.apply_alignment(hits_x=hits_chunk['x_dut_%d' % dut_index][selection],
                                                                                                                                                                 hits_y=hits_chunk['y_dut_%d' % dut_index][selection],
                                                                                                                                                                 hits_z=hits_chunk['z_dut_%d' % dut_index][selection],
                                                                                                                                                                 dut_index=dut_index,
                                                                                                                                                                 alignment=alignment,
                                                                                                                                                                 inverse=inverse)
                            if not no_z:
                                hits_chunk['z_dut_%d' % dut_index][selection] = hit_z

                    hits_aligned_table.append(hits_chunk)

    logging.debug('File with newly aligned hits %s', output_hit_aligned_file)


def alignment(input_track_candidates_file, input_alignment_file, n_pixels, pixel_size, ignore_duts=None, max_iterations=10, use_n_tracks=100000, plot_result=True, chunk_size=1000000):
    ''' This function does an alignment of the DUTs and sets translation and rotation values for all DUTs.
    The reference DUT defines the global coordinate system position at 0, 0, 0 and should be well in the beam and not heavily rotated.

    To solve the chicken-and-egg problem that a good dut alignment needs hits belonging to one track, but good track finding needs a good dut alignment this
    function work only on already prealigned hits belonging to one track. Thus this function can be called only after track finding.

    These steps are done
    1. Take the found tracks and revert the prealignment
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
    ignore_duts : iterable
        the duts that are not taken in a fit. Needed to exclude small planes / very bad working planes from the fit
        to make a good alignment possible
    max_iterations : number
        Maximum iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations)
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

    def calculate_translation_alignment(track_candidates_file, iteration, output_pdf, total_residual, plot_title_praefix='', include_duts=None):
        ''' Main function that fits tracks, calculates the residuals, deduces rotation and translation values from the residuals
        and applies the new alignment to the track hits. The alignment result is scored as a combined
        residual value of all planes in x and y. '''

        apply_alignment(input_hit_file=track_candidates_file[:-3] + '_no_align_tmp.h5',  # Always apply alignment to starting file
                        input_alignment=input_alignment_file,
                        output_hit_aligned_file=track_candidates_file[:-3] + '_no_align_%d_tmp.h5' % iteration,
                        chunk_size=chunk_size)

        # Step 2: Fit tracks for all DUTs
        logging.info('= Alignment step 2 / iteration %d: Fit tracks for all DUTs =', iteration)
        fit_tracks(input_track_candidates_file=track_candidates_file[:-3] + '_no_align_%d_tmp.h5' % iteration,
                   input_alignment_file=input_alignment_file,
                   output_tracks_file=track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % iteration,
                   track_quality=1,
                   include_duts=include_duts,
                   ignore_duts=ignore_duts)  # In the first iteration there is no alignment

        # Step 3: Calculate the residuals for each DUT
        logging.info('= Alignment step 3 / iteration %d: Calculate the residuals for each DUT =', iteration)
        calculate_residuals(input_tracks_file=track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % iteration,
                            input_alignment_file=input_alignment_file,
                            output_residuals_file=track_candidates_file[:-3] + '_residuals_%d_tmp.h5' % iteration,
                            n_pixels=n_pixels,
                            pixel_size=pixel_size,
                            output_pdf=False,
                            chunk_size=chunk_size)

        # Step 4: Deduce rotations from the residuals
        logging.info('= Alignment step 4 / iteration %d: Deduce rotations and translations from the residuals =', iteration)
        alignment_parameters, new_total_residual = _analyze_residuals(track_candidates_file[:-3] + '_residuals_%d_tmp.h5' % iteration,
                                                                      output_pdf,
                                                                      n_duts,
                                                                      plot_title_praefix=plot_title_praefix)

        if total_residual and new_total_residual > total_residual:
            logging.info('Best alignment found')
            return -1  # Abort is signaled with total_residual < 0
        else:
            total_residual = new_total_residual

        # Step 5: Set rotation information in alignment file
        logging.info('= Alignment step 5/ iteration %d: Set rotation information in alignment file =', iteration)
        geometry_utils.store_alignment_parameters(input_alignment_file,
                                                  alignment_parameters,
                                                  mode='relative')

        return total_residual

    # Open the prealignment and create the alignment info (at the beginning only the z position is set)
    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        alignment_parameters = _create_alignment_array(n_duts)
        alignment_parameters['translation_z'] = prealignment['z']

    geometry_utils.store_alignment_parameters(input_alignment_file,
                                              alignment_parameters,
                                              mode='absolute')

    # Step 0: Reduce the number of tracks to increase the calculation time
    logging.info('= Alignment step 0: Reduce number of tracks to %d =', use_n_tracks)
    dut_selection = 0
    for i in range(n_duts):
        if ignore_duts and i in ignore_duts:
            continue
        dut_selection |= (1 << i)
    track_quality = 1
    data_selection.select_hits(hit_file=input_track_candidates_file,
                               max_hits=use_n_tracks,
                               track_quality=(dut_selection << (track_quality * 8)),
                               track_quality_mask=(dut_selection << (track_quality * 8)),
                               chunk_size=chunk_size)
    reduced_track_candidates_file = input_track_candidates_file[:-3] + '_reduced.h5'  # Rename to reduced file

    # Step 1: Take the found tracks and revert the prealignment
    logging.info('= Alignment step 1: Revert prealignment =')
    apply_alignment(input_hit_file=reduced_track_candidates_file,
                    input_alignment=input_alignment_file,  # Revert prealignent
                    output_hit_aligned_file=reduced_track_candidates_file[:-3] + '_no_align_tmp.h5',
                    inverse=True,
                    force_prealignment=True,
                    chunk_size=chunk_size)

    total_residual = 0
    # Stage N: Repeat alignment with constrained residuals until total residual does not decrease anymore
    for iteration in range(0, max_iterations):
        total_residual = calculate_translation_alignment(track_candidates_file=reduced_track_candidates_file,
                                                         iteration=iteration,
                                                         output_pdf=False,
                                                         total_residual=total_residual)

        if total_residual < 0:  # Abort is checked within calculate_translation_alignment function and signaled with total_residual < 0
            break

        if iteration >= max_iterations:
            raise RuntimeError('Did not converge to good solution in %d iterations. Increase max_iterations', iteration)

    # Plot final result
    if plot_result:
        logging.info('= Alignment step 6: Plot final result =')
        with PdfPages(os.path.join(os.path.dirname(os.path.realpath(input_track_candidates_file)), 'Alignment.pdf')) as output_pdf:
            # Revert prealignment
            apply_alignment(input_hit_file=input_track_candidates_file,
                            input_alignment=input_alignment_file,
                            output_hit_aligned_file=input_track_candidates_file[:-3] + '_no_prealignment_tmp.h5',
                            inverse=True,  # Revert prealignment
                            force_prealignment=True,
                            chunk_size=chunk_size)
            # Apply final alignment result
            apply_alignment(input_hit_file=input_track_candidates_file[:-3] + '_no_prealignment_tmp.h5',
                            input_alignment=input_alignment_file,
                            output_hit_aligned_file=input_track_candidates_file[:-3] + '_final_tmp.h5',
                            # reset_z=True,
                            chunk_size=chunk_size)
            fit_tracks(input_track_candidates_file=input_track_candidates_file[:-3] + '_final_tmp.h5',
                       input_alignment_file=input_alignment_file,
                       output_tracks_file=input_track_candidates_file[:-3] + '_tracks_final_tmp.h5',
                       track_quality=1,
                       ignore_duts=ignore_duts)
            calculate_residuals(input_tracks_file=input_track_candidates_file[:-3] + '_tracks_final_tmp.h5',
                                input_alignment_file=input_alignment_file,
                                output_residuals_file=input_track_candidates_file[:-3] + '_residuals_final_tmp.h5',
                                n_pixels=n_pixels,
                                pixel_size=pixel_size,
                                output_pdf=False,
                                chunk_size=chunk_size)
            _analyze_residuals(residuals_file_h5=input_track_candidates_file[:-3] + '_residuals_final_tmp.h5',
                               output_fig=output_pdf,
                               n_duts=n_duts,
                               plot_title_praefix='Unconstrained residual after alignment')
            os.remove(input_track_candidates_file[:-3] + '_final_tmp.h5')
            os.remove(input_track_candidates_file[:-3] + '_tracks_final_tmp.h5')
            os.remove(input_track_candidates_file[:-3] + '_tracks_final_tmp.pdf')
            os.remove(input_track_candidates_file[:-3] + '_residuals_final_tmp.h5')

    # Delete temporary files
    for i in range(iteration + 1):
        os.remove(reduced_track_candidates_file[:-3] + '_no_align_%d_tmp.h5' % i)
        os.remove(reduced_track_candidates_file[:-3] + '_tracks_%d_tmp.h5' % i)
        os.remove(reduced_track_candidates_file[:-3] + '_tracks_%d_tmp.pdf' % i)
        os.remove(reduced_track_candidates_file[:-3] + '_residuals_%d_tmp.h5' % i)
    os.remove(reduced_track_candidates_file)

    logging.info('Alignment finished successfully!')


# Helper functions for the alignment. Not to be used directly.

def _get_rotation_from_residual_fit(m_xx, m_xy, m_yx, m_yy):
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
    return np.zeros((n_duts,), dtype=description)


def _analyze_residuals(residuals_file_h5, output_fig, n_duts, plot_title_praefix=''):
    ''' Take the residual plots and deduce rotation and translation angles from them '''
    alignment_parameters = _create_alignment_array(n_duts)

    total_residual = 0  # Sum of all residuals to judge the overall alignment

    with tb.open_file(residuals_file_h5) as in_file_h5:
        for dut_index in range(n_duts):
                # Global residuals
            hist_node = in_file_h5.get_node('/ResidualsX_DUT%d' % dut_index)
            # Calculate bins from edges
            edges_x = np.linspace(hist_node._v_attrs.x_edges[0], hist_node._v_attrs.x_edges[-1], num=hist_node[:].shape[0] + 1)
            x = edges_x + np.diff(edges_x)[0]  # Center bins
            x = x[:-1]  # Get rid of extra bin
            y = hist_node[:]
            mu_x = analysis_utils.get_mean_from_histogram(y, edges_x[:-1])
            std = analysis_utils.get_rms_from_histogram(y, edges_x[:-1])
            coeff_x, var_matrix = None, None
            try:
                coeff_x, var_matrix = curve_fit(analysis_utils.gauss, edges_x[:-1], y, p0=[np.max(y), mu_x, std])
            except RuntimeError:  # Fit failed
                pass

            total_residual = np.sqrt(np.square(total_residual) + np.square(std))  # Maybe better to use sigma from gauss fit?

            alignment_parameters[dut_index]['correlation_x'] = std

            if output_fig is not False:
                plot_utils.plot_residuals(histogram=(y, edges_x),
                                          fit=coeff_x,
                                          fit_errors=var_matrix,
                                          title='Residuals for DUT %d' % dut_index,
                                          x_label='X residual [um]',
                                          output_fig=output_fig)

            hist_node = in_file_h5.get_node('/ResidualsY_DUT%d' % dut_index)
            # Calculate bins from edges
            edges_x = np.linspace(hist_node._v_attrs.x_edges[0], hist_node._v_attrs.x_edges[-1], num=hist_node[:].shape[0] + 1)
            x = edges_x + np.diff(x)[0]  # Center bins
            x = x[:-1]  # Get rid of extra bin
            y = hist_node[:]
            mu_y = analysis_utils.get_mean_from_histogram(y, x)
            std = analysis_utils.get_rms_from_histogram(y, x)
            coeff_x, var_matrix = None, None
            try:
                coeff_y, var_matrix = curve_fit(analysis_utils.gauss, x, y, p0=[np.max(y), mu_y, std])
            except RuntimeError:  # Fit failed
                pass

            total_residual = np.sqrt(np.square(total_residual) + np.square(std))  # Maybe better to use sigma from gauss fit?

            alignment_parameters[dut_index]['correlation_y'] = std

            if output_fig is not False:
                plot_utils.plot_residuals(histogram=(y, edges_x),
                                          fit=coeff_y,
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
            y_sigma = n_hits.sum() / n_hits
            n_hits_threshold = np.percentile(hist_x_residual_x[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
            x_fit = x[n_hits > n_hits_threshold]
            y_fit = y[n_hits > n_hits_threshold]
            y_fit_sigma = y_sigma[n_hits > n_hits_threshold]
            popt, pcov = curve_fit(line, x_fit, y_fit, sigma=y_fit_sigma, absolute_sigma=False)  # Fit straight line
            if output_fig is not False:
                plot_utils.plot_position_residuals((hist_x_residual_x, x_edges, y_edges), x, y, yerr=y_sigma, x_label='X position [um]', y_label='X residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
            m_xx = popt[1]

            hist_y_residual_y = in_file_h5.get_node('/YResidualsY_DUT%d' % dut_index)
            x_edges = hist_y_residual_y._v_attrs.x_edges
            y_edges = hist_y_residual_y._v_attrs.y_edges
            x = np.linspace(x_edges[0], x_edges[-1], num=hist_y_residual_y.shape[0])
            values = np.linspace(y_edges[0], y_edges[-1], num=hist_y_residual_y.shape[1])
            n_hits = hist_y_residual_y[:].sum(axis=1)
            y = get_median(hist_y_residual_y, values)
            y_sigma = n_hits.sum() / n_hits
            n_hits_threshold = np.percentile(hist_y_residual_y[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
            x_fit = x[n_hits > n_hits_threshold]
            y_fit = y[n_hits > n_hits_threshold]
            y_fit_sigma = y_sigma[n_hits > n_hits_threshold]
            popt, pcov = curve_fit(line, x_fit, y_fit, sigma=y_fit_sigma, absolute_sigma=False)  # Fit straight line
            if output_fig is not False:
                plot_utils.plot_position_residuals((hist_y_residual_y, x_edges, y_edges), x, y, yerr=y_sigma, x_label='Y position [um]', y_label='Y residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
            m_yy = popt[1]

            hist_x_residual_y = in_file_h5.get_node('/XResidualsY_DUT%d' % dut_index)
            x_edges = hist_x_residual_y._v_attrs.x_edges
            y_edges = hist_x_residual_y._v_attrs.y_edges
            x = np.linspace(x_edges[0], x_edges[-1], num=hist_x_residual_y.shape[0])
            values = np.linspace(y_edges[0], y_edges[-1], num=hist_x_residual_y.shape[1])
            n_hits = hist_x_residual_y[:].sum(axis=1)
            y = get_median(hist_x_residual_y, values)
            y_sigma = n_hits.sum() / n_hits
            n_hits_threshold = np.percentile(hist_x_residual_y[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
            x_fit = x[n_hits > n_hits_threshold]
            y_fit = y[n_hits > n_hits_threshold]
            y_fit_sigma = y_sigma[n_hits > n_hits_threshold]
            popt, pcov = curve_fit(line, x_fit, y_fit, sigma=y_fit_sigma, absolute_sigma=False)  # Fit straight line
            if output_fig is not False:
                plot_utils.plot_position_residuals((hist_x_residual_y, x_edges, y_edges), x, y, yerr=y_sigma, x_label='X position [um]', y_label='Y residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
            m_xy = popt[1]

            hist_y_residual_x = in_file_h5.get_node('/YResidualsX_DUT%d' % dut_index)
            x_edges = hist_y_residual_x._v_attrs.x_edges
            y_edges = hist_y_residual_x._v_attrs.y_edges
            x = np.linspace(x_edges[0], x_edges[-1], num=hist_y_residual_x.shape[0])
            values = np.linspace(y_edges[0], y_edges[-1], num=hist_y_residual_x.shape[1])
            n_hits = hist_y_residual_x[:].sum(axis=1)
            y = get_median(hist_y_residual_x, values)
            y_sigma = n_hits.sum() / n_hits
            n_hits_threshold = np.percentile(hist_y_residual_x[:].sum(axis=1), 30)  # Simple threshold, to get rid of the 30% of lowest entry bins
            x_fit = x[n_hits > n_hits_threshold]
            y_fit = y[n_hits > n_hits_threshold]
            y_fit_sigma = y_sigma[n_hits > n_hits_threshold]
            popt, pcov = curve_fit(line, x_fit, y_fit, sigma=y_fit_sigma, absolute_sigma=False)  # Fit straight line
            if output_fig is not False:
                plot_utils.plot_position_residuals((hist_y_residual_x, x_edges, y_edges), x, y, yerr=y_sigma, x_label='Y position [um]', y_label='X residual [um]', title=plot_title_praefix + ', DUT%d' % dut_index, output_fig=output_fig, fit=(popt, pcov))
            m_yx = popt[1]

            alpha, beta, gamma = _get_rotation_from_residual_fit(m_xx, m_xy, m_yx, m_yy)

            alignment_parameters[dut_index]['DUT'] = dut_index
            alignment_parameters[dut_index]['alpha'] = alpha
            alignment_parameters[dut_index]['beta'] = beta
            alignment_parameters[dut_index]['gamma'] = gamma

            alignment_parameters[dut_index]['translation_x'] = -mu_x
            alignment_parameters[dut_index]['translation_y'] = -mu_y

    return alignment_parameters, total_residual


def align_z(input_track_candidates_file, input_alignment_file, use_n_tracks=100000, ignore_duts=None, chunk_size=10000000):
    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        alignment = in_file_h5.root.Alignment[:]
#         z_positions = alignment['translation_z']
        n_duts = prealignment.shape[0]

    def _reduce_data(input_track_candidates_file, dut_selection, use_n_tracks):  # Step 0: Reduce the number of tracks to increase the calculation time
        logging.info('= Z step 0: Reduce number of tracks to %d =', use_n_tracks)
        track_quality = 1
        data_selection.select_hits(hit_file=input_track_candidates_file,
                                   max_hits=use_n_tracks,
                                   track_quality=(dut_selection << (track_quality * 8)),
                                   track_quality_mask=(dut_selection << (track_quality * 8)),
                                   chunk_size=chunk_size)
        reduced_track_candidates_file = input_track_candidates_file[:-3] + '_reduced.h5'  # Rename to reduced file
        with tb.open_file(reduced_track_candidates_file) as in_file_h5:
            return in_file_h5.root.TrackCandidates[:]

    def _fit_tracks(track_candidates, dut_selection, z_corrections):  # Step 1: Fit tracks, do not use fit track method to reduce the overhead and save time
        # Prepare track hits array to be fitted
        n_fit_duts = bin(dut_selection).count("1")
        index, n_tracks = 0, track_candidates['event_number'].shape[0]  # Index of tmp track hits array
        track_hits = np.zeros((n_tracks, n_fit_duts, 3))
        for dut_index in range(n_duts):  # Fill index loop of new array
            if (1 << dut_index) & dut_selection == (1 << dut_index):  # True if DUT is used in fit
                xyz = np.column_stack((track_candidates['x_dut_%s' % dut_index], track_candidates['y_dut_%s' % dut_index], track_candidates['z_dut_%s' % dut_index])) + z_corrections[dut_index]
                track_hits[:, index, :] = xyz
                index += 1
    
        # Split data and fit on all available cores
        n_slices = cpu_count()
        slice_length = np.ceil(1. * n_tracks / n_slices).astype(np.int32)
        slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
        pool = Pool(n_slices)
        results = pool.map(track_analysis._fit_tracks_loop, slices)
        pool.close()
        pool.join()
    
        offsets = np.concatenate([i[0] for i in results])  # Merge offsets from all cores in results
        slopes = np.concatenate([i[1] for i in results])  # Merge slopes from all cores in results

        return track_hits, offsets, slopes
    
    def _minimize_me(z_correction, actual_dut):
        dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z'] + z_correction])
        rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
                                                         beta=alignment[actual_dut]['beta'],
                                                         gamma=alignment[actual_dut]['gamma'])
        basis_global = rotation_matrix.T.dot(np.eye(3))
        dut_plane_normal = basis_global[2]
         
        intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                          line_directions=slopes,
                                                                          position_plane=dut_position,
                                                                          normal_plane=dut_plane_normal)
         
        return np.sqrt(np.square(np.std(track_hits[:, actual_dut, 0] - intersections[:, 0])) + np.square(np.std(track_hits[:, actual_dut, 1] - intersections[:, 1])))

    dut_selection = 0
    for i in range(n_duts):
        if ignore_duts and i in ignore_duts:
            continue
        dut_selection |= (1 << i)

    dut_selection = (1 << (n_duts - 1) | 1)
    z_corrections = np.zeros(n_duts)
    track_candidates = _reduce_data(input_track_candidates_file, dut_selection, use_n_tracks=use_n_tracks)
    
    residuals = np.zeros(n_duts)
    
    _, offsets, slopes = _fit_tracks(track_candidates, dut_selection, z_corrections=z_corrections)



    for actual_dut in range(n_duts):
        residuals_plt =[]
        #_, offsets, slopes = _fit_tracks(track_candidates, dut_selection | (1 << actual_dut), z_corrections=z_corrections)
        for z_correction in range(-1000000, 1000000, 10000):
            dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z'] + z_correction])
            rotation_matrix = geometry_utils.rotation_matrix(alpha=0,
                                                             beta=0,
                                                             gamma=0)
            basis_global = rotation_matrix.T.dot(np.eye(3))
            dut_plane_normal = basis_global[2]
                
            intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                              line_directions=slopes,
                                                                              position_plane=dut_position,
                                                                              normal_plane=dut_plane_normal)
                
            residual = np.sqrt(np.square(np.std(track_candidates['x_dut_%s' % actual_dut] - intersections[:, 0])) + np.square(np.std(track_candidates['y_dut_%s' % actual_dut] - intersections[:, 1])))
                
            residuals_plt.append(residual)
                
        plt.title('DUT%d' % actual_dut)
        plt.plot(range(-1000000, 1000000, 10000), residuals_plt)
        plt.show()
    
    
#     for actual_dut in range(n_duts):
#         for i in range(100):
#             track_hits, offsets, slopes = _fit_tracks(track_candidates, dut_selection, z_corrections=z_corrections)
# #             res = minimize_scalar(_minimize_me, bounds=(-50000, 50000), args=(actual_dut), method='bounded')
# #             z_corrections[actual_dut] = -res.x
# #             residuals[actual_dut] = res.fun
# #             print 'z_corrections', z_corrections 
# #             print 'residuals', residuals
# #             print i, residuals.sum()
#         
#             for actual_dut in range(n_duts):
#                 residuals_plt =[]
#                 for z_correction in range(-10000, 10000, 100):
#                     dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z']]) + z_correction
#                     rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
#                                                                      beta=alignment[actual_dut]['beta'],
#                                                                      gamma=alignment[actual_dut]['gamma'])
#                     basis_global = rotation_matrix.T.dot(np.eye(3))
#                     dut_plane_normal = basis_global[2]
#                         
#                     intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
#                                                                                       line_directions=slopes,
#                                                                                       position_plane=dut_position,
#                                                                                       normal_plane=dut_plane_normal)
#                         
#                     residual = np.sqrt(np.square(np.std(track_hits[:, actual_dut, 0] - intersections[:, 0])) + np.square(np.std(track_hits[:, actual_dut, 1] - intersections[:, 1])))
#                         
#                     residuals_plt.append(residual)
#                         
#                 plt.title('DUT%d' % actual_dut)
#                 plt.plot(range(-10000, 10000, 100), residuals_plt)
#                 plt.show()
# #             
        
            

    def total_residuals(z_corrections, track_hits, offsets, slopes):
        residual_sum = 0
    
        # Step 2: Calcualte residual
        for actual_dut in range(n_duts):
            dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z']]) + z_corrections[actual_dut]
            rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
                                                             beta=alignment[actual_dut]['beta'],
                                                             gamma=alignment[actual_dut]['gamma'])
            basis_global = rotation_matrix.T.dot(np.eye(3))
            dut_plane_normal = basis_global[2]
            
            intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                              line_directions=slopes,
                                                                              position_plane=dut_position,
                                                                              normal_plane=dut_plane_normal)
            
            residual = np.sqrt(np.square(np.std(track_hits[:, actual_dut, 0] - intersections[:, 0])) + np.square(np.std(track_hits[:, actual_dut, 1] - intersections[:, 1])))
            residual_sum += residual
          
        track_hits[:, :, 2] -= z_corrections[np.newaxis, :]  # Reverse change for next call
        print 'residual_sum', residual_sum
        return residual_sum
    
    def reconstruct_z(track_hits):
        ''' Reconstructs the real rotation by changing the rotation angles until the correlation
        in column/row can be described best by a straight line with the slope of 1.

        Parameters:
        ----------
        column_0, row_0, column_1, row_1 : np.array
            The column/row positions of DUT0 and other DUT
        x0 : iterable
            Start parameters from pre alignment or measurement (alpha_0, beta_0, gamma_0, alpha_1, beta_1, gamma_1, ...).
            E.g. for 2 DUTs: x0 = (0., 0., 0., pi, 0.01, 0.)
        errors : iterable of iterable
            Maximum error for the angles of DUT 1. Sets the limit of the possible reconstructed rotation angles and should be choosen
            carefully. E.g.: ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))
            If None +- 5 degree maximum error from the starting values x0 is assumed

        Returns:
        -------

        Tuple with (alha, beta, gamma) rotation angles
        '''
#         bounds = []
#         for i in range(3):  # Loop over angles
#             bounds.append((x0[i] - errors[i], x0[i] + errors[i]))
#         
#         
#         
        n_duts = track_hits.shape[1]
        bounds = [(-5000, 5000)] * n_duts
# 
#         niter = 10  # Iterations of basinhopping, so far did never really change the result
# 
#         progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=niter, term_width=80)
#         progress_bar.start()
# 
#         def callback(x, f, accept):  # Callback called after each basinhopping step, used to show a progressbar
#             progress_bar.update(progress_bar.currval + 1)
# 
#         result = basinhopping(func=total_residuals,
#                               #T=0.5,
#                               niter=niter,
#                               x0=np.zeros(n_duts),
#                               callback=callback,
#                               minimizer_kwargs={'args': track_hits,
#                                                 'method': 'SLSQP',
#                                                 'bounds': bounds
#                                                 })
# 
#         progress_bar.finish()
        
#         result = minimize(fun=total_residuals,
#                 x0=np.zeros(n_duts),
#                 args=(track_hits),
#                 bounds=bounds,
#                 #method='SLSQP',
#                 options={'eps':1000})
# 
#         return result.x
# 
#     minimm = 1000000
#     
#     reconstruct_z(track_hits)

#     a = np.zeros(n_duts)
# 
#     for i in range(-10000, 10000, 1000):
#         a[0] = i
#         print i, total_residuals(a, track_hits)
    

#     #print total_residuals(track_hits, z_corrections=np.zeros_like(z_positions))
#     print total_residuals(track_hits, z_corrections=np.ones_like((z_positions)) * 100. )
#     print total_residuals(track_hits, z_corrections=np.zeros_like(z_positions))

#         y, edges_x = np.histogram(difference[:, 0], bins=1000)
#         
#         mu_x = analysis_utils.get_mean_from_histogram(y, edges_x[:-1])
#         std = analysis_utils.get_rms_from_histogram(y, edges_x[:-1])
#         coeff_x, var_matrix = curve_fit(analysis_utils.gauss, edges_x[:-1], y, p0=[np.max(y), mu_x, std])
#         plt.bar(edges_x[:-1], y, width=np.diff(edges_x)[0])
#         x = np.arange(edges_x[0], edges_x[-1])
#         plt.plot(x, analysis_utils.gauss(x, *coeff_x))
#         print 'sigma_x', std
#         plt.show()
#         
#         y, edges_x = np.histogram(difference[:, 1], bins=1000)
#         
#         mu_x = analysis_utils.get_mean_from_histogram(y, edges_x[:-1])
#         std = analysis_utils.get_rms_from_histogram(y, edges_x[:-1])
#         coeff_x, var_matrix = curve_fit(analysis_utils.gauss, edges_x[:-1], y, p0=[np.max(y), mu_x, std])
#         plt.bar(edges_x[:-1], y, width=np.diff(edges_x)[0])
#         x = np.arange(edges_x[0], edges_x[-1])
#         plt.plot(x, analysis_utils.gauss(x, *coeff_x))
#         print 'sigma_y', std
#         plt.show()
        
#         dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z']])
#         rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[fit_dut]['alpha'],
#                                                          beta=alignment[fit_dut]['beta'],
#                                                          gamma=alignment[fit_dut]['gamma'])
#         basis_global = rotation_matrix.T.dot(np.eye(3))  # TODO: why transposed?
#         dut_plane_normal = basis_global[2]
#         
#          actual_offsets = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
#                                                                           line_directions=slopes,
#                                                                           position_plane=dut_position,
#                                                                           normal_plane=dut_plane_normal)
#         
#         transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=alignment[actual_dut]['translation_x'],
#                                                                                      y=alignment[actual_dut]['translation_y'],
#                                                                                      z=alignment[actual_dut]['translation_z'],
#                                                                                      alpha=alignment[actual_dut]['alpha'],
#                                                                                      beta=alignment[actual_dut]['beta'],
#                                                                                      gamma=alignment[actual_dut]['gamma'])
# 
#         hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_transformation_matrix(x=track_hits[:, actual_dut, :][0],
#                                                                                            y=track_hits[:, actual_dut, :][1],
#                                                                                            z=track_hits[:, actual_dut, :][2],
#                                                                                            transformation_matrix=transformation_matrix)
# 
# 
# 
#         intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_transformation_matrix(x=results[1][0],
#                                                                                                                       y=results[1][1],
#                                                                                                                       z=results[1][2],
#                                                                                                                       transformation_matrix=transformation_matrix)
# 
#         if not np.allclose(hit_z_local, 0) or not np.allclose(intersection_z_local, 0):
#             logging.error('Hit z position = %s and z intersection %s', str(hit_z_local[:3]), str(intersection_z_local[:3]))
#             raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')
