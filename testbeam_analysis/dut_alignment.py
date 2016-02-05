''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''

import logging
import re
import progressbar
import tables as tb
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit, minimize_scalar
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis import analysis_utils
from testbeam_analysis import plot_utils


def correlate_hits(hit_files, alignment_file, fraction=1, event_range=0):
    '''Histograms the hit column (row)  of two different devices on an event basis. If the hits are correlated a line should be seen.
    The correlation is done very simple. Not all hits of the first device are correlated with all hits of the second device. This is sufficient
    as long as you do not have too many hits per event.

    Parameters
    ----------
    input_file : pytables file
    fraction: int
        Take only every fraction-th hit to save time. Not needed with low statistics runs.
    alignment_file : pytables file
        Output file with the correlation data
    event_range: int or iterable
        select events for which the correlation is done
        if 0: select all events
        if int: select first int events
        if list of int (length 2): select events from first list item to second list item
    '''
    logging.info('=== Correlate the position of %d DUTs ===', len(hit_files))
    with tb.open_file(alignment_file, mode="w") as out_file_h5:
        for index, hit_file in enumerate(hit_files):
            with tb.open_file(hit_file, 'r') as in_file_h5:
                # Set event selection
                event_range = [event_range, ] if not isinstance(event_range, list) else event_range
                if len(event_range) == 2:
                    event_start, event_end = event_range[0], event_range[1]
                else:
                    event_start = 0
                    if event_range[0] == 0:
                        event_end = None
                    else:
                        event_end = event_range[0]

                hit_table = in_file_h5.root.Hits[event_start:event_end:fraction]
                if index == 0:
                    first_reference = pd.DataFrame({'event_number': hit_table[:]['event_number'], 'column_ref': hit_table[:]['column'], 'row_ref': hit_table[:]['row'], 'tot_ref': hit_table[:]['charge']})
                    n_col_reference, n_row_reference = np.amax(hit_table[:]['column']), np.amax(hit_table[:]['row'])
                else:
                    logging.info('Correlate detector %d with detector %d', index, 0)
                    dut = pd.DataFrame({'event_number': hit_table[:]['event_number'], 'column_dut': hit_table[:]['column'], 'row_dut': hit_table[:]['row'], 'tot_dut': hit_table[:]['charge']})
                    df = first_reference.merge(dut, how='left', on='event_number')
                    df.dropna(inplace=True)
                    n_col_dut, n_row_dut = np.amax(hit_table[:]['column']), np.amax(hit_table[:]['row'])
                    # Correlation of x against x and y against y
                    col_corr = analysis_utils.hist_2d_index(df['column_dut'] - 1, df['column_ref'] - 1, shape=(n_col_dut, n_col_reference))
                    row_corr = analysis_utils.hist_2d_index(df['row_dut'] - 1, df['row_ref'] - 1, shape=(n_row_dut, n_row_reference))
#                     TODO: implement rotated devices
#                     # Correlation of x against y and y against x (should be constant exepct the sensor is rotated)
#                     row_col_corr = analysis_utils.hist_2d_index(df['row_dut'] - 1, df['column_ref'] - 1, shape=(n_row_dut, n_col_reference))
#                     col_row_corr = analysis_utils.hist_2d_index(df['column_dut'] - 1, df['row_ref'] - 1, shape=(n_col_dut, n_row_reference))
                    out_col = out_file_h5.createCArray(out_file_h5.root, name='CorrelationColumn_%d_0' % index, title='Column Correlation between DUT %d and %d' % (index, 0), atom=tb.Atom.from_dtype(col_corr.dtype), shape=col_corr.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    out_row = out_file_h5.createCArray(out_file_h5.root, name='CorrelationRow_%d_0' % index, title='Row Correlation between DUT %d and %d' % (index, 0), atom=tb.Atom.from_dtype(row_corr.dtype), shape=row_corr.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    out_col.attrs.filenames = [str(hit_files[0]), str(hit_files[index])]
                    out_row.attrs.filenames = [str(hit_files[0]), str(hit_files[index])]
                    out_col[:] = col_corr
                    out_row[:] = row_corr


def align_hits(correlation_file, pixel_size, alignment_file, output_pdf, fit_offset_cut=[(10. / 10, 10. / 10)], fit_error_cut=[(100. / 1000, 100. / 1000)], show_plots=False):
    '''Takes the correlation histograms, determines useful ranges with valid data, fits the correlations and stores the correlation parameters. With the
    correlation parameters one can calculate the hit position of each DUT in the master reference coordinate system. The fits are
    also plotted.

    Parameters
    ----------
    correlation_file : pytbales file
        The input file with the correlation histograms
    alignment_file : pytables file
        The output file for correlation data.
    combine_bins : int
        Rebin the alignment histograms to get better statistics
    fit_offset_cut : float / list, (column, row)
        Omit channels where the distance to the fit is > fit_offset_cut
        Happens e.g. if there is no clear correlation due to noise, insufficient statistics
        If given a list of floats use one list item for each DUT
    fit_error_cut : float / list, (column, row)
        Omit channels where the fit has an error > fit_error_cut
        Happens e.g. if there is no clear correlation due to noise, insufficient statistics
        If given a list of floats use one list item for each DUT
    output_pdf : pdf file name object
    '''
    logging.info('=== Align hit coordinates ===')

    def gauss(x, *p):
        A, mu, sigma, offset = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset

    fit_offset_cut = [fit_offset_cut, ] if not isinstance(fit_offset_cut, list) else fit_offset_cut
    fit_error_cut = [fit_error_cut, ] if not isinstance(fit_error_cut, list) else fit_error_cut

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(correlation_file, mode="r+") as in_file_h5:
            n_nodes = sum(1 for _ in enumerate(in_file_h5.root))  # Determine number of nodes, is there a better way?
            n_duts = int(n_nodes / 2 + 1)
            result = np.zeros(shape=(n_nodes,), dtype=[('dut_x', np.uint8), ('dut_y', np.uint8), ('c0', np.float), ('c0_error', np.float), ('c1', np.float), ('c1_error', np.float), ('c2', np.float), ('c2_error', np.float), ('sigma', np.float), ('sigma_error', np.float)])
            for node_index, node in enumerate(in_file_h5.root):
                try:
                    result[node_index]['dut_x'], result[node_index]['dut_y'] = int(re.search(r'\d+', node.name).group()), node.name[-1:]
                except AttributeError:
                    continue
                logging.info('Align %s', node.name)

                # TODO: allow pixel_size for each DUT or one size for all DUTs
                if 'Col' in node.title:  # differ between column and row for sensors with rectangular pixels
                    pixel_length, pixel_length_ref = pixel_size[node_index + 1][0], pixel_size[0][0]
                else:
                    pixel_length, pixel_length_ref = pixel_size[node_index - n_duts + 2][1], pixel_size[0][1]

                data = node[:]

                # Start values for fitting
                mus = np.argmax(data, axis=1)
                As = np.max(data, axis=1)

                # Fit result arrays have -1 for bad fit
                mean_fitted = np.array([-1. for _ in range(data.shape[0])])
                mean_error_fitted = np.array([-1. for _ in range(data.shape[0])])
                sigma_fitted = np.array([-1. for _ in range(data.shape[0])])
                chi2 = np.array([-1. for _ in range(data.shape[0])])

                # Loop over all row/row or column/column slices and fit a gaussian to the profile
                # Get values with highest correlation for alignment fit
                # Do this with channel indices, later convert to um
                x_hist_fit = np.arange(1.5, data.shape[1] + 1.5)  # Set bin centers as data points

                for index in np.arange(data.shape[0]):
                    p0 = [As[index], mus[index], 1., 0.]
                    try:
                        coeff, var_matrix = curve_fit(gauss, x_hist_fit, data[index, :], p0=p0)
                        mean_fitted[index] = coeff[1]
                        mean_error_fitted[index] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
                        sigma_fitted[index] = coeff[2]
                        if index == data.shape[0] / 2:
                            plot_utils.plot_correlation_fit(x_hist_fit, data[index, :], coeff, var_matrix, 'DUT 0 at DUT %s = %d' % (result[node_index]['dut_x'], index), node.title, output_fig)
                    except RuntimeError:
                        pass

                # Unset invalid data
                mean_fitted[~np.isfinite(mean_fitted)] = -1
                mean_error_fitted[~np.isfinite(mean_error_fitted)] = -1

                # Convert fit results to um for alignment fit
                mean_fitted *= pixel_length_ref
                mean_error_fitted = pixel_length_ref * mean_error_fitted

                # Fit selected data with a straight line 3 times to remove outliers
                selected_data = np.arange(data.shape[0])
                x_align_fit = np.arange(1.5, data.shape[0] + 1.5)

                for i in range(3):
                    f = lambda x, c0, c1: c0 + c1 * x
                    if not np.any(selected_data):
                        raise RuntimeError('The cuts are too tight, there is no point to fit. Release cuts and rerun alignment.')
                    if len(fit_offset_cut) == 1 and len(fit_error_cut) == 1:  # Use same fit_offset_cut and fit_error_cut values for all fits
                        offset_limit, error_limit = fit_offset_cut[0][0] if 'Col' in node.title else fit_offset_cut[0][1], fit_error_cut[0][0] if 'Col' in node.title else fit_error_cut[0][1]
                    else:  # Use different fit_offset_cut and fit_error_cut values for every fit
                        offset_limit, error_limit = fit_offset_cut[node_index][0] if 'Col' in node.title else fit_offset_cut[node_index - n_duts + 1][1], fit_error_cut[node_index][0] if 'Col' in node.title else fit_error_cut[node_index - n_duts + 1][1]

                    fit, pcov = curve_fit(f, pixel_length * x_align_fit[selected_data], mean_fitted[selected_data])
                    fit_fn = np.poly1d(fit[::-1])
                    offset = fit_fn(pixel_length * x_align_fit) - mean_fitted
                    selected_data = np.where(np.logical_and(mean_error_fitted > 1e-3, np.logical_and(np.abs(offset) < offset_limit, mean_error_fitted < error_limit)))[0]
                    if show_plots and np.any(selected_data):
                        plot_utils.plot_alignments(data, selected_data, pixel_length, mean_fitted, fit_fn, mean_error_fitted, offset, result, node_index, i, node.title, offset_limit, error_limit)

                # Refit with higher polynomial
                # Use results from straight line fit as start values for last fit
                g = lambda x, c0, c1, c2: c0 + c1 * x + c2 * x ** 2
                fit, pcov = curve_fit(g, pixel_length * x_align_fit[selected_data], mean_fitted[selected_data], sigma=mean_error_fitted[selected_data], absolute_sigma=True, p0=[fit[0], fit[1], 0.])
                fit_fn = np.poly1d(fit[::-1])

                # Calculate mean sigma (is somewhat a residual) and its error and store the actual data in result array
                mean_sigma = pixel_length_ref * np.mean(np.array(sigma_fitted)[selected_data])
                mean_sigma_error = pixel_length_ref * np.std(np.array(sigma_fitted)[selected_data]) / np.sqrt(np.array(sigma_fitted)[selected_data].shape[0])

                # Write fit results to array
                result[node_index]['c0'], result[node_index]['c0_error'] = fit[0], np.absolute(pcov[0][0]) ** 0.5
                result[node_index]['c1'], result[node_index]['c1_error'] = fit[1], np.absolute(pcov[1][1]) ** 0.5
                result[node_index]['c2'], result[node_index]['c2_error'] = fit[2], np.absolute(pcov[2][2]) ** 0.5

                result[node_index]['sigma'], result[node_index]['sigma_error'] = mean_sigma, mean_sigma_error

                # Plot selected data with fit
                plot_utils.plot_alignment_fit(data, selected_data, pixel_length, mean_fitted, fit_fn, fit, pcov, chi2, mean_error_fitted, result, node_index, i, node.title, show_plots, output_fig)

            with tb.open_file(alignment_file, mode="w") as out_file_h5:
                try:
                    result_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', description=result.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    result_table.append(result)
                except tb.exceptions.NodeError:
                    logging.warning('Correlation table exists already. Do not create new.')


def merge_cluster_data(cluster_files, alignment_file, tracklets_file, pixel_size, max_index=None):
    '''Takes the cluster from all cluster files and merges them into one big table onto the event number.
    Empty entries are signaled with charge = 0. The position is referenced from the correlation data to the first plane.
    Function uses easily several GB of RAM. If memory errors occur buy a better PC or chunk this function.

    Parameters
    ----------
    cluster_files : list of pytables files
        Files with cluster data
    alignment_file : pytables files
        The file with the correlation data
    track_candidates_file : pytables files
    max_index : int
        Merge only given number of cluster data
    '''
    logging.info('=== Merge cluster to tracklets ===')

    with tb.open_file(alignment_file, mode="r") as in_file_h5:
        correlation = in_file_h5.root.Alignment[:]

    # Calculate a new event number index to map the cluster files of all planes into it
    rows = 0
    for cluster_file in cluster_files:
        with tb.open_file(cluster_file, mode='r') as in_file_h5:
            rows = max(rows, in_file_h5.root.Cluster.cols.event_number[-1])
    common_bin_count = None
    for cluster_file in cluster_files:
            with tb.open_file(cluster_file, mode='r') as in_file_h5:
                bin_count = np.bincount(in_file_h5.root.Cluster.cols.event_number[:], minlength=rows + 1)
                if common_bin_count is None:
                    common_bin_count = bin_count
                else:
                    common_bin_count = np.maximum(common_bin_count, bin_count)
    # http://stackoverflow.com/questions/22671192/inverse-of-numpys-bincount-function
    p = np.cumsum(common_bin_count)
    i = np.zeros(p[-1], np.int)
    np.add.at(i, p[:-1], 1)
    common_event_number = np.cumsum(i).astype(np.int64)
    # this does the same:
    # common_event_number = np.repeat(np.arange(common_bin_count.size), common_bin_count)

#     # Calculate a event number index to map the cluster of all files to
#     common_event_number_2 = None
#     for cluster_file in cluster_files:
#         with tb.open_file(cluster_file, mode='r') as in_file_h5:
#             common_event_number_2 = in_file_h5.root.Cluster[:]['event_number'] if common_event_number_2 is None else analysis_utils.get_max_events_in_both_arrays(common_event_number_2, in_file_h5.root.Cluster[:]['event_number'])

    # Create result array description, depends on the number of DUTs
    description = [('event_number', np.int64)]
    for index, _ in enumerate(cluster_files):
        description.append(('column_dut_%d' % index, np.float))
    for index, _ in enumerate(cluster_files):
        description.append(('row_dut_%d' % index, np.float))
    for index, _ in enumerate(cluster_files):
        description.append(('charge_dut_%d' % index, np.float))
    description.extend([('track_quality', np.uint32), ('n_tracks', np.uint8)])

    # Merge the cluster data from different DUTs into one table
    with tb.open_file(tracklets_file, mode='w') as out_file_h5:
        tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracklets', description=np.zeros((1,), dtype=description).dtype, title='Tracklets', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        tracklets_array = np.zeros((common_event_number.shape[0],), dtype=description)
        for index, cluster_file in enumerate(cluster_files):
            logging.info('Add cluster file ' + str(cluster_file))
            with tb.open_file(cluster_file, mode='r') as in_file_h5:
                actual_cluster = analysis_utils.map_cluster(common_event_number, in_file_h5.root.Cluster[:])
                selection = actual_cluster['mean_column'] != 0  # correct only hits, 0 is no hit
                actual_mean_column = pixel_size[index][0] * actual_cluster['mean_column'][selection]  # Convert channel indices to um
                actual_mean_row = pixel_size[index][1] * actual_cluster['mean_row'][selection]  # Convert channel indices to um
                if index == 0:  # Position corrections are normalized to the first reference
                    c0 = np.array([0., 0.])
                    c1 = np.array([1., 1.])
                    c2 = np.array([0., 0.])
                else:
                    c0 = correlation[correlation['dut_x'] == index]['c0']
                    c1 = correlation[correlation['dut_x'] == index]['c1']
                    c2 = correlation[correlation['dut_x'] == index]['c2']

                # Apply alignment information
                tracklets_array['column_dut_%d' % index][selection] = (c2[0] * actual_mean_column ** 2 + c1[0] * actual_mean_column + c0[0])
                tracklets_array['row_dut_%d' % index][selection] = (c2[1] * actual_mean_row ** 2 + c1[1] * actual_mean_row + c0[1])
                tracklets_array['charge_dut_%d' % index][selection] = actual_cluster['charge'][selection]

        np.nan_to_num(tracklets_array)
        tracklets_array['event_number'] = common_event_number
        if max_index:
            tracklets_array = tracklets_array[:max_index]
        tracklets_table.append(tracklets_array)


def fix_event_alignment(tracklets_files, tracklets_corr_file, alignment_file, error=3., n_bad_events=100, n_good_events=10, correlation_search_range=20000, good_events_search_range=100):
    '''Description

    Parameters
    ----------
    tracklets_file: pytables file
        Input file with original Tracklet data
    tracklets_corr_file: pyables_file
        Output file for corrected Tracklet data
    alignment_file: pytables file
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
    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] / 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    logging.info('=== Fix event alignment ===')

    with tb.open_file(tracklets_files, mode="r") as in_file_h5:
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


def optimize_hit_alignment(tracklets_files, alignment_file, fraction=10):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    aligment_file : string
        Input file name with alignment data
    use_fraction : float
        Use only every fraction-th hit for the alignment correction. For speed up. 1 means all hits are used
    '''
    logging.info('=== Optimize hit alignment ===')
    with tb.open_file(tracklets_files, mode="r+") as in_file_h5:
        particles = in_file_h5.root.Tracklets[:]
        with tb.open_file(alignment_file, 'r+') as alignment_file_h5:
            alignment_data = alignment_file_h5.root.Alignment[:]
            n_duts = alignment_data.shape[0] / 2
            for table_column in in_file_h5.root.Tracklets.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    actual_dut = int(table_column[-1:])
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Optimize alignment for % s', table_column)
                    particle_selection = particles[::fraction][np.logical_and(particles[::fraction][ref_dut_column] > 0, particles[::fraction][table_column] > 0)]  # only select events with hits in both DUTs
                    difference = particle_selection[ref_dut_column] - particle_selection[table_column]
                    selection = np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)  # select all hits from events with hits in both DUTs
                    particles[table_column][selection] += np.median(difference)
                    # Shift values by deviation from median
                    if 'col' in table_column:
                        alignment_data['c0'][actual_dut - 1] -= np.median(difference)
                    else:
                        alignment_data['c0'][actual_dut + n_duts - 1] -= np.median(difference)
            # Store corrected/new alignment table after deleting old table
            alignment_file_h5.removeNode(alignment_file_h5.root, 'Alignment')
            result_table = alignment_file_h5.create_table(alignment_file_h5.root, name='Alignment', description=alignment_data.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            result_table.append(alignment_data)
        in_file_h5.removeNode(in_file_h5.root, 'Tracklets')
        corrected_tracklets_table = in_file_h5.create_table(in_file_h5.root, name='Tracklets', description=particles.dtype, title='Tracklets', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        corrected_tracklets_table.append(particles)


def check_hit_alignment(tracklets_file, output_pdf, combine_n_hits=100000, correlated_only=False):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf : pdf file name object
    combine_n_hits : int
        The number of events to combine for the hit position check
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('=== Check hit alignment ===')
    with tb.open_file(tracklets_file, mode="r") as in_file_h5:
        with PdfPages(output_pdf) as output_fig:
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


def align_z(track_candidates_file, alignment_file, output_pdf, z_positions=None, track_quality=1, max_tracks=3, warn_at=0.5):
    '''Minimizes the squared distance between track hit and measured hit by changing the z position.
    In a perfect measurement the function should be minimal at the real DUT position. The tracks is given
    by the first and last reference hit. A track quality cut is applied to all cuts first.

    Parameters
    ----------
    track_candidates_file : pytables file
    alignment_file : pytables file
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
        with tb.open_file(track_candidates_file, mode='r') as in_file_h5:
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

    with tb.open_file(alignment_file, mode='r+') as out_file_h5:
        try:
            z_table_out = out_file_h5.createTable(out_file_h5.root, name='Zposition', description=results.dtype, title='Relative z positions of the DUTs without references', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            z_table_out.append(results)
        except tb.NodeError:
            logging.warning('Z position are do already exist. Do not overwrite.')

    z_positions_rec = np.add(([0.] + results[:]['z_position_row'].tolist() + [1.]), ([0.] + results[:]['z_position_column'].tolist() + [1.])) / 2.

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
