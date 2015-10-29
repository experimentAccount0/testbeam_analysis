"""This script does a full test beam analysis. As an input raw data files with a trigger number from one
run are expected. This script does in-RAM calculations on multiple cores in parallel. 12 GB of free RAM and 8 cores are recommended.
The analysis flow is (also mentioned again in the __main__ section):
- Do for each DUT in parallel
  - Create a hit tables from the raw data
  - Align the hit table event number to the trigger number to be able to correlate hits in time
  - Cluster the hit table
- Create hit position correlations from the hit maps and store the arrays
- Take the correlation arrays and extract an offset/slope aligned to the first DUT
- Merge the cluster tables from all DUTs to one big cluster table and reference the cluster positions to the reference (DUT0) position
- Find tracks
- Align the DUT positions in z (optional)
- Fit tracks (very simple, fit straight line without hit correlations taken into account)
- Calculate residuals
- Create efficiency / distance maps
"""

import logging
import progressbar
import re
import numpy as np
from math import sqrt
import pandas as pd
import tables as tb
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit, minimize_scalar

from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis.hit_clusterizer import HitClusterizer
from testbeam_analysis.clusterizer import data_struct
from testbeam_analysis import analysis_utils
from testbeam_analysis import plot_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def remove_hot_pixels(data_file, threshold=6.):
    '''Std. analysis of a hit table. Clusters are created.

    Parameters
    ----------
    data_file : pytables file
    threshold : number
        The threshold when the pixel is removed given in sigma distance from the median occupancy.
    '''
    logging.info('Remove hot pixels in %s', data_file)
    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_hot_pixel.h5', 'w') as out_file_h5:
            hits = input_file_h5.root.Hits[:]
            col, row = hits['column'], hits['row']
            max_row = np.amax(row)
            occupancy = analysis_utils.hist_2d_index(col - 1, row - 1, shape=(np.amax(col), max_row))
            noisy_pixels = np.where(occupancy > np.median(occupancy) + np.std(occupancy) * threshold)
            plot_utils.plot_noisy_pixel(occupancy, noisy_pixels, threshold, filename=data_file[:-3] + '_hot_pixel.pdf')
            logging.info('Remove %d hot pixels in %s', noisy_pixels[0].shape[0], data_file)

            # Select not noisy pixels
            noisy_pix_1d = (noisy_pixels[0] + 1) * max_row + (noisy_pixels[1] + 1)  # map 2d array (col, row) to 1d array to increase selection speed
            hits_1d = hits['column'].astype(np.uint32) * max_row + hits['row']  # astype needed, otherwise silently assuming np.uint16 (VERY BAD NUMPY!)
            hits = hits[np.in1d(hits_1d, noisy_pix_1d, invert=True)]

            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=hits.dtype, title='Selected not noisy hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hit_table_out.append(hits)


def cluster_hits(data_file, max_x_distance=3, max_y_distance=3):
    '''Std. analysis of a hit table. Clusters are created.

    Parameters
    ----------
    data_file : pytables file
    output_file : pytables file
    '''

    logging.info('Cluster hits in %s', data_file)

    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_cluster.h5', 'w') as output_file_h5:
            hits = input_file_h5.root.Hits[:]
            clusterizer = HitClusterizer(np.amax(hits['column']), np.amax(hits['row']))
            clusterizer.set_x_cluster_distance(max_x_distance)  # cluster distance in columns
            clusterizer.set_y_cluster_distance(max_y_distance)  # cluster distance in rows
            clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames
            cluster = np.zeros_like(hits, dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
            clusterizer.set_cluster_info_array(cluster)  # tell the array to be filled
            clusterizer.add_hits(hits)
            cluster = cluster[:clusterizer.get_n_clusters()]
            cluster_table_description = data_struct.ClusterInfoTable().columns.copy()
            cluster_table_out = output_file_h5.createTable(output_file_h5.root, name='Cluster', description=cluster_table_description, title='Clustered hits', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            cluster_table_out.append(cluster)


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
    logging.info('Correlate the position of %d DUTs', len(hit_files))
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
                    col_corr = analysis_utils.hist_2d_index(df['column_dut'] - 1, df['column_ref'] - 1, shape=(n_col_dut, n_col_reference))
                    row_corr = analysis_utils.hist_2d_index(df['row_dut'] - 1, df['row_ref'] - 1, shape=(n_row_dut, n_row_reference))
                    out_col = out_file_h5.createCArray(out_file_h5.root, name='CorrelationColumn_%d_0' % index, title='Column Correlation between DUT %d and %d' % (index, 0), atom=tb.Atom.from_dtype(col_corr.dtype), shape=col_corr.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    out_row = out_file_h5.createCArray(out_file_h5.root, name='CorrelationRow_%d_0' % index, title='Row Correlation between DUT %d and %d' % (index, 0), atom=tb.Atom.from_dtype(row_corr.dtype), shape=row_corr.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    out_col.attrs.filenames = [str(hit_files[0]), str(hit_files[index])]
                    out_row.attrs.filenames = [str(hit_files[0]), str(hit_files[index])]
                    out_col[:] = col_corr
                    out_row[:] = row_corr


def align_hits(correlation_file, alignment_file, output_pdf, fit_offset_cut=[(10. / 10, 10. / 10)], fit_error_cut=[(100. / 1000, 100. / 1000)], show_plots=False):
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
    combine_bins : float
        Omit channels where the number of hits is < no_data_cut * mean channel hits
        Happens e.g. if the device is not fully illuminated
    fit_error_cut : float / iterable
        Omit channels where the fit has an error > fit_error_cut
        Happens e.g. if there is no clear correlation due to noise, insufficient statistics
        If given a list of floats use one list item for each DUT
    output_pdf : pdf file name object
    '''
    logging.info('Align hit coordinates')

    def gauss(x, *p):
        A, mu, sigma, offset = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset

    fit_offset_cut = [fit_offset_cut, ] if not isinstance(fit_offset_cut, list) else fit_offset_cut
    fit_error_cut = [fit_error_cut, ] if not isinstance(fit_error_cut, list) else fit_error_cut

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(correlation_file, mode="r+") as in_file_h5:
            n_nodes = sum(1 for _ in enumerate(in_file_h5.root))  # Determine number of nodes, is there a better way?
            result = np.zeros(shape=(n_nodes,), dtype=[('dut_x', np.uint8), ('dut_y', np.uint8), ('c0', np.float), ('c0_error', np.float), ('c1', np.float), ('c1_error', np.float), ('c2', np.float), ('c2_error', np.float), ('sigma', np.float), ('sigma_error', np.float), ('description', np.str_, 40)])
            for node_index, node in enumerate(in_file_h5.root):
                try:
                    result[node_index]['dut_x'], result[node_index]['dut_y'] = int(re.search(r'\d+', node.name).group()), node.name[-1:]
                except AttributeError:
                    continue
                logging.info('Align %s', node.name)

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
                channel_indices = np.arange(data.shape[0])
                for index in channel_indices:
                    p0 = [As[index], mus[index], 1., 1.]
                    try:
                        x = np.arange(data.shape[1])
                        coeff, var_matrix = curve_fit(gauss, x, data[index, :], p0=p0)
                        mean_fitted[index] = coeff[1]
                        mean_error_fitted[index] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
                        sigma_fitted[index] = coeff[2]
                        if index == data.shape[0] / 2:
                            plot_utils.plot_correlation_fit(x, data[index, :], coeff, var_matrix, 'DUT 0 at DUT %s = %d' % (result[node_index]['dut_x'], index), node.title, output_fig)
                    except RuntimeError:
                        pass

                mean_error_fitted = np.abs(mean_error_fitted)

                # Fit data with a straight line 3 times to remove outliers
                selected_data = range(data.shape[0])
                for i in range(3):
                    f = lambda x, c0, c1: c0 + c1 * x
                    if not np.any(selected_data):
                        raise RuntimeError('The cuts are too tight, there is no point to fit. Release cuts and rerun alignment.')
                    index = 0
                    if len(fit_offset_cut) == 1 and len(fit_error_cut) == 1:  # use same fit_offset_cut and fit_error_cut values for all fits
                        offset_limit, error_limit = fit_offset_cut[0][0] if 'Col' in node.title else fit_offset_cut[0][1], fit_error_cut[0][0] if 'Col' in node.title else fit_error_cut[0][1]
                    else:  # use different fit_offset_cut and fit_error_cut values for every fit
                        index = node_index % len(fit_offset_cut)
                        offset_limit, error_limit = fit_offset_cut[index][0] if 'Col' in node.title else fit_offset_cut[index][1], fit_error_cut[index][0] if 'Col' in node.title else fit_error_cut[index][1]

                    fit, pcov = curve_fit(f, np.arange(data.shape[0])[selected_data], mean_fitted[selected_data])
                    fit_fn = np.poly1d(fit[::-1])
                    offset = fit_fn(np.arange(data.shape[0])) - mean_fitted
                    selected_data = np.where(np.logical_and(mean_error_fitted > 1e-3, np.logical_and(np.abs(offset) < offset_limit, mean_error_fitted < error_limit)))
                    if show_plots:
                        plot_utils.plot_alignments(data, selected_data, mean_fitted, fit_fn, mean_error_fitted, offset, result, node_index, i, node.title)

                # Refit with higher polynomial
                g = lambda x, c0, c1, c2, c3: c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
                fit, pcov = curve_fit(g, np.arange(data.shape[0])[selected_data], mean_fitted[selected_data], sigma=mean_error_fitted[selected_data], absolute_sigma=True)
                fit_fn = np.poly1d(fit[::-1])

                # Calculate mean sigma (is somewhat a residual) and store the actual data in result array
                mean_sigma = np.mean(np.array(sigma_fitted)[selected_data])
                mean_sigma_error = np.std(np.array(sigma_fitted)[selected_data]) / np.sqrt(channel_indices[selected_data].shape[0])
                result[node_index]['c0'], result[node_index]['c0_error'] = fit[0], np.absolute(pcov[0][0]) ** 0.5
                result[node_index]['c1'], result[node_index]['c1_error'] = fit[1], np.absolute(pcov[1][1]) ** 0.5
                result[node_index]['c2'], result[node_index]['c2_error'] = fit[2], np.absolute(pcov[2][2]) ** 0.5
                result[node_index]['sigma'], result[node_index]['sigma_error'] = mean_sigma, mean_sigma_error
                result[node_index]['description'] = node.title

                # Plot selected data with fit
                plot_utils.plot_alignment_fit(data, selected_data, mean_fitted, fit_fn, fit, pcov, chi2, mean_error_fitted, offset, result, node_index, i, node.title, output_fig)

            with tb.open_file(alignment_file, mode="w") as out_file_h5:
                try:
                    result_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', description=result.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    result_table.append(result)
                except tb.exceptions.NodeError:
                    logging.warning('Correlation table exists already. Do not create new.')


def merge_cluster_data(cluster_files, alignment_file, tracklets_file, max_index=None):
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
    logging.info('Merge cluster to tracklets')
    with tb.open_file(alignment_file, mode="r") as in_file_h5:
        correlation = in_file_h5.root.Alignment[:]

    # Calculate a event number index to map the cluster of all files to
    common_event_number = None
    for cluster_file in cluster_files:
        with tb.open_file(cluster_file, mode='r') as in_file_h5:
            common_event_number = in_file_h5.root.Cluster[:]['event_number'] if common_event_number is None else analysis_utils.get_max_events_in_both_arrays(common_event_number, in_file_h5.root.Cluster[:]['event_number'])

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
                selection = actual_cluster['mean_column'] != 0
                actual_mean_column = actual_cluster['mean_column'][selection]  # correct only hits, 0 is no hit
                actual_mean_row = actual_cluster['mean_row'][selection]  # correct only hits, 0 is no hit
                if index == 0:  # Position corrections are normalized to the first reference
                    c0 = np.array([0., 0.])
                    c1 = np.array([1., 1.])
                    c2 = np.array([0., 0.])
                else:
                    c0 = correlation[correlation['dut_x'] == index]['c0']
                    c1 = correlation[correlation['dut_x'] == index]['c1']
                    c2 = correlation[correlation['dut_x'] == index]['c2']

                tracklets_array['column_dut_%d' % index][selection] = c2[0] * actual_mean_column ** 2 + c1[0] * actual_mean_column + c0[0]
                tracklets_array['row_dut_%d' % index][selection] = c2[1] * actual_mean_row ** 2 + c1[1] * actual_mean_row + c0[1]
                tracklets_array['charge_dut_%d' % index][selection] = actual_cluster['charge'][selection]

#         np.nan_to_num(tracklets_array)
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

    # get alignment errors
    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] / 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    logging.info('Fix event alignment')

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
                particles_corrected['column_dut_0'] = ref_column
                particles_corrected['row_dut_0'] = ref_row
                particles_corrected['charge_dut_0'] = ref_charge  # copy values that have not been changed
                particles_corrected['n_tracks'] = particles['n_tracks']
                particles_corrected[table_column] = column
                particles_corrected['row_dut_' + table_column[-1]] = row
                particles_corrected['charge_dut_' + table_column[-1]] = charge

                correlation_index = np.where(correlated == 1)[0]

                particles_corrected['track_quality'][correlation_index] |= (1 << (24 + int(table_column[-1])))

        with tb.open_file(tracklets_corr_file, mode="w") as out_file_h5:
            try:
                out_file_h5.root.Tracklets._f_remove(recursive=True, force=False)
                logging.warning('Overwrite old corrected Tracklets file')
            except tb.NodeError:
                logging.info('Create new corrected Tracklets file')

            correction_out = out_file_h5.create_table(out_file_h5.root, name='Tracklets', description=in_file_h5.root.Tracklets.description, title='Corrected Tracklets data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            correction_out.append(particles_corrected)


def optimize_hit_alignment(tracklets_files, alignment_file, use_fraction=0.1):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    aligment_file : string
        Input file name with alignment data
    use_fraction : float
        The fraction of hits to used for the alignment correction. For speed up. 1 means all hits are used
    '''
    logging.info('Optimize hit alignment')
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
                    every_nth_hit = int(1. / use_fraction)
                    particle_selection = particles[::every_nth_hit][np.logical_and(particles[::every_nth_hit][ref_dut_column] > 0, particles[::every_nth_hit][table_column] > 0)]  # only select events with hits in both DUTs
                    difference = particle_selection[ref_dut_column] - particle_selection[table_column]
                    selection = np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)  # select all hits from events with hits in both DUTs
                    particles[table_column][selection] += np.median(difference)
                    # Change linear offset of alignment
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


def check_hit_alignment(tracklets_files, output_pdf, combine_n_hits=100000, correlated_only=False):
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
    logging.info('Check hit alignment')
    with tb.open_file(tracklets_files, mode="r") as in_file_h5:
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

                        actual_median, actual_mean, actual_rms = np.median(difference), np.mean(difference), np.std(difference)
                        alignment.append(np.median(np.abs(difference)))
                        correlation.append(difference.shape[0] * 100. / combine_n_hits)

                        median.append(actual_median)
                        mean.append(actual_mean)
                        std.append(actual_rms)

                        plot_utils.plot_hit_alignment('Aligned position difference for events %d - %d' % (index, index + combine_n_hits), difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_fig, bins=100)
                        progress_bar.update(index)
                    plot_utils.plot_hit_alignment_2(in_file_h5, combine_n_hits, median, mean, correlation, alignment, output_fig)
                    progress_bar.finish()


def find_tracks(tracklets_file, alignment_file, track_candidates_file, pixel_size):
    '''Takes first DUT track hit and tries to find matching hits in subsequent DUTs.
    The output is the same array with resorted hits into tracks. A track quality is given to
    be able to cut on good tracks.
    This function is slow since the main loop happens in Python (< 1e5 tracks / second) but does the track finding
    loop on all cores in parallel (_find_tracks_loop()).

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    alignment_file : string
        File containing the alignment information
    track_candidates_file : string
        Output file name for track candidate array
    '''
    logging.info('Build tracks from tracklets')

    # calculate pixel dimension ratio (col/row) for hit distance
    pixel_ratio = float(pixel_size[0]) / pixel_size[1]

    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] / 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    with tb.open_file(tracklets_file, mode='r') as in_file_h5:
        tracklets = in_file_h5.root.Tracklets
        n_duts = sum(['column' in col for col in tracklets.dtype.names])
        n_slices = cpu_count() - 1
        n_tracks = tracklets.nrows
        slice_length = n_tracks / n_slices
        slices = [tracklets[i:i + slice_length] for i in range(0, n_tracks, slice_length)]

        pool = Pool(n_slices)  # let all cores work the array
        arg = [(one_slice, n_duts, column_sigma, row_sigma, pixel_ratio) for one_slice in slices]  # FIXME: slices are not aligned at event numbers, up to n_slices * 2 tracks are found wrong
        results = pool.map(_function_wrapper_find_tracks_loop, arg)
        result = np.concatenate(results)
        pool.close()
        pool.join()

        with tb.open_file(track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=in_file_h5.root.Tracklets.description, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            track_candidates.append(result)


def find_tracks_corr(tracklets_file, alignment_file, track_candidates_file):
    '''Takes first DUT track hit and tries to find matching hits in subsequent DUTs.
    Works on corrected Tracklets file, is identical to the find_tracks function.
    The output is the same array with resorted hits into tracks. A track quality is given to
    be able to cut on good tracks.
    This function is slow since the main loop happens in Python (< 1e5 tracks / second)
    but does the track finding loop on all cores in parallel (_find_tracks_loop()).
    Does the same as find_tracks function but works on a corrected TrackCandidates file (optimize_track_aligment)

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    alignment_file : string
        File containing the alignment information
    track_candidates_file : string
        Output file name for track candidate array
    '''
    logging.info('Build tracks from tracklets')

    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] / 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    with tb.open_file(tracklets_file, mode='r') as in_file_h5:
        tracklets = in_file_h5.root.TrackCandidates
        n_duts = sum(['column' in col for col in tracklets.dtype.names])
        n_slices = cpu_count() - 1
        n_tracks = tracklets.nrows
        slice_length = n_tracks / n_slices
        slices = [tracklets[i:i + slice_length] for i in range(0, n_tracks, slice_length)]

        pool = Pool(n_slices)  # let all cores work the array
        arg = [(one_slice, n_duts, column_sigma, row_sigma) for one_slice in slices]  # FIXME: slices are not aligned at event numbers, up to n_slices * 2 tracks are found wrong
        results = pool.map(_function_wrapper_find_tracks_loop, arg)
        result = np.concatenate(results)
        pool.close()
        pool.join()

# _find_tracks_loop_compiled = jit((numpy_support.from_dtype(tracklets.dtype)[:], types.int32, types.float64, types.float64), nopython=True)(_find_tracks_loop)  # maybe in 1 year this will help, when numba works with structured arrays
#         _find_tracks_loop(tracklets, correlations, n_duts, column_sigma, row_sigma)
#         result = tracklets

        with tb.open_file(track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=in_file_h5.root.TrackCandidates.description, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            track_candidates.append(result)


def optimize_track_alignment(trackcandidates_file, alignment_file, use_fraction=1, correlated_only=False):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).
    Does the same as optimize_hit_alignment but works on TrackCandidates file.
    If optimize_track_aligment is used track quality can change and must be calculated again from corrected data (use find_tracks_corr).


    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    alignment_file : string
        Input file with alignment data
    use_fraction : float
        The fraction of hits to used for the alignment correction. For speed up. 1 means all hits are used
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('Optimize track alignment')
    with tb.open_file(trackcandidates_file, mode="r+") as in_file_h5:
        particles = in_file_h5.root.TrackCandidates[:]
        with tb.open_file(alignment_file, 'r+') as alignment_file_h5:
            alignment_data = alignment_file_h5.root.Alignment[:]
            n_duts = alignment_data.shape[0] / 2
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    actual_dut = int(table_column[-1:])
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Optimize alignment for % s', table_column)
                    every_nth_hit = int(1. / use_fraction)
                    particle_selection = particles[::every_nth_hit][np.logical_and(particles[::every_nth_hit][ref_dut_column] > 0, particles[::every_nth_hit][table_column] > 0)]  # only select events with hits in both DUTs
                    difference = particle_selection[ref_dut_column] - particle_selection[table_column]
                    selection = np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)  # select all hits from events with hits in both DUTs
                    particles[table_column][selection] += np.median(difference)
                    # Change linear offset of alignmet
                    if 'col' in table_column:
                        alignment_data['c0'][actual_dut - 1] -= np.median(difference)
                    else:
                        alignment_data['c0'][actual_dut + n_duts - 1] -= np.median(difference)
            # Store corrected/new alignment table after deleting old table
            alignment_file_h5.removeNode(alignment_file_h5.root, 'Alignment')
            result_table = alignment_file_h5.create_table(alignment_file_h5.root, name='Alignment', description=alignment_data.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            result_table.append(alignment_data)
        in_file_h5.removeNode(in_file_h5.root, 'TrackCandidates')
        corrected_trackcandidates_table = in_file_h5.create_table(in_file_h5.root, name='TrackCandidates', description=particles.dtype, title='TrackCandidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        corrected_trackcandidates_table.append(particles)


def check_track_alignment(trackcandidates_files, output_pdf, combine_n_hits=10000000, correlated_only=False):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).
    Does the same as check_hit_alignment but works on TrackCandidates file.

    Parameters
    ----------
    trackcandidates_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf : pdf file name object
    combine_n_hits : int
        The number of events to combine for the hit position check
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('Check TrackCandidates Alignment')
    with tb.open_file(trackcandidates_files, mode="r") as in_file_h5:
        with PdfPages(output_pdf) as output_fig:
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    median, mean, std, alignment, correlation = [], [], [], [], []
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Check alignment for % s', table_column)
                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.TrackCandidates.shape[0], term_width=80)
                    progress_bar.start()
                    for index in range(0, in_file_h5.root.TrackCandidates.shape[0], combine_n_hits):
                        particles = in_file_h5.root.TrackCandidates[index:index + combine_n_hits:5]  # take every 10th hit
                        particles = particles[np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)]  # only select events with hits in both DUTs
                        if correlated_only is True:
                            particles = particles[particles['track_quality'] & (1 << (24 + int(table_column[-1]))) == (1 << (24 + int(table_column[-1])))]
                        if particles.shape[0] == 0:
                            logging.warning('No correlation for dut %s and events %d - %d', table_column, index, index + combine_n_hits)
                            median.append(-1)
                            mean.append(-1)
                            std.append(-1)
                            alignment.append(0)
                            correlation.append(0)
                            continue
                        difference = particles[:][ref_dut_column] - particles[:][table_column]

                        actual_median, actual_mean = np.median(difference), np.mean(difference)
                        alignment.append(np.median(np.abs(difference)))
                        correlation.append(difference.shape[0] * 100. / combine_n_hits)
                        plot_utils.plot_hit_alignment('Aligned position difference', difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_fig, bins=100)

                        progress_bar.update(index)

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
    logging.info('Find relative z-position')

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


def fit_tracks(track_candidates_file, tracks_file, z_positions, fit_duts=None, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], max_tracks=1, track_quality=1, pixel_size=(250, 50), output_pdf=None, use_correlated=False):
    '''Fits a line through selected DUT hits for selected DUTs. The selection criterion for the track candidates to fit is the track quality and the maximum number of hits per event.
    The fit is done for specified DUTs only (fit_duts). This DUT is then not included in the fit (include_duts). Bad DUTs can be always ignored in the fit (ignore_duts).

    Parameters
    ----------
    track_candidates_file : string
        file name with the track candidates table
    tracks_file : string
        file name of the created track file having the track table
    z_position : iterable
        the positions of the devices in z in cm
    fit_duts : iterable
        the duts to fit tracks for. If None all duts are used
    ignore_duts : iterable
        the duts that are not taken in a fit. Needed to exclude bad planes from track fit. Also included Duts are ignored!
    include_duts : iterable
        the relative dut positions of dut to use in the track fit. The position is relative to the actual dut the tracks are fitted for
        e.g. actual track fit dut = 2, include_duts = [-3, -2, -1, 1] means that duts 0, 1, 3 are used for the track fit
    max_tracks : int
        only events with tracks <= max tracks are taken
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    pixel_size : iterable, (x dimensions, y dimension)
        the size in um of the pixels, needed for chi2 calculation
    output_pdf : pdf file name object
        if None plots are printed to screen
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''

    def create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts):
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('column_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('row_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('charge_dut_%d' % index, np.float))
        for dimension in range(3):
            description.append(('offset_%d' % dimension, np.float))
        for dimension in range(3):
            description.append(('slope_%d' % dimension, np.float))
        description.extend([('track_chi2', np.uint32), ('track_quality', np.uint32), ('n_tracks', np.uint8)])

        tracks_array = np.zeros((n_tracks,), dtype=description)
        tracks_array['event_number'] = good_track_candidates['event_number']
        tracks_array['track_quality'] = good_track_candidates['track_quality']
        tracks_array['n_tracks'] = good_track_candidates['n_tracks']
        for index in range(n_duts):
            tracks_array['column_dut_%d' % index] = good_track_candidates['column_dut_%d' % index]
            tracks_array['row_dut_%d' % index] = good_track_candidates['row_dut_%d' % index]
            tracks_array['charge_dut_%d' % index] = good_track_candidates['charge_dut_%d' % index]
        for dimension in range(3):
            tracks_array['offset_%d' % dimension] = offsets[:, dimension]
            tracks_array['slope_%d' % dimension] = slopes[:, dimension]
        tracks_array['track_chi2'] = chi2s

        return tracks_array

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(track_candidates_file, mode='r') as in_file_h5:
            with tb.open_file(tracks_file, mode='w') as out_file_h5:
                track_candidates = in_file_h5.root.TrackCandidates[:]
                n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                fit_duts = fit_duts if fit_duts else range(n_duts)
                for fit_dut in fit_duts:  # loop over the duts to fit the tracks for
                    logging.info('Fit tracks for DUT %d', fit_dut)

                    # Select track candidates
                    dut_selection = 0  # duts to be used in the fit
                    quality_mask = 0  # masks duts to check track quality for
                    for include_dut in include_duts:  # calculate mask to select DUT hits for fitting
                        if fit_dut + include_dut < 0 or ((ignore_duts and fit_dut + include_dut in ignore_duts) or fit_dut + include_dut >= n_duts):
                            continue
                        if include_dut >= 0:
                            dut_selection |= ((1 << fit_dut) << include_dut)
                        else:
                            dut_selection |= ((1 << fit_dut) >> abs(include_dut))

                        quality_mask = dut_selection | (1 << fit_dut)

                    if bin(dut_selection).count("1") < 2:
                        logging.warning('Insufficient track hits to do fit (< 2). Omit DUT %d', fit_dut)
                        continue

                    good_track_selection = np.logical_and((track_candidates['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8)), track_candidates['n_tracks'] <= max_tracks)

                    print 'Lost due to normal cuts', good_track_selection.shape[0] - np.sum(good_track_selection)

                    if use_correlated:  # reduce track selection to only correlated duts
                        good_track_selection &= (track_candidates['track_quality'] & (quality_mask << 24) == (quality_mask << 24))
                        print 'Lost due to correlated cuts', good_track_selection.shape[0] - np.sum(track_candidates['track_quality'] & (quality_mask << 24) == (quality_mask << 24))

                    good_track_candidates = track_candidates[good_track_selection]

                    # Prepare track hits array to be fitted
                    n_fit_duts = bin(dut_selection).count("1")
                    index, n_tracks = 0, good_track_candidates['event_number'].shape[0]  # index of tmp track hits array
                    track_hits = np.zeros((n_tracks, n_fit_duts, 3))
                    for dut_index in range(0, n_duts):  # fill index loop of new array
                        if (1 << dut_index) & dut_selection == (1 << dut_index):  # true if dut is used in fit
                            xyz = np.column_stack((good_track_candidates['column_dut_%s' % dut_index], good_track_candidates['row_dut_%s' % dut_index], np.repeat(z_positions[dut_index], n_tracks)))
                            track_hits[:, index, :] = xyz
                            index += 1

                    # Split data and fit on all available cores
                    n_slices = cpu_count() - 1
                    slice_length = np.ceil(1. * n_tracks / n_slices).astype(np.int32)

                    slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
                    pool = Pool(n_slices)
                    arg = [(one_slice, pixel_size) for one_slice in slices]  # FIXME: slices are not aligned at event numbers, up to n_slices * 2 tracks are found wrong
                    results = pool.map(_function_wrapper_fit_tracks_loop, arg)
                    pool.close()
                    pool.join()
                    del track_hits

                    # Store results
                    offsets = np.concatenate([i[0] for i in results])  # merge offsets from all cores in results
                    slopes = np.concatenate([i[1] for i in results])  # merge slopes from all cores in results
                    chi2s = np.concatenate([i[2] for i in results])  # merge chi2 from all cores in results
                    tracks_array = create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts)
                    tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    tracklets_table.append(tracks_array)

                    plot_utils.plot_track_chi2(chi2s, fit_dut, output_fig)


def calculate_residuals(tracks_file, z_positions, pixel_size=(250, 50), use_duts=None, max_chi2=None, output_pdf=None):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    z_position : iterable
        the positions of the devices in z in cm
    use_duts : iterable
        the duts to calculate residuals for. If None all duts are used
    max_chi2 : int
        USe only converged fits (cut on chi2)
    output_pdf : pdf file name
        if None plots are printed to screen
    '''
    logging.info('Calculate residuals')

    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    output_fig = PdfPages(output_pdf) if output_pdf else None

    with tb.open_file(tracks_file, mode='r') as in_file_h5:
        for node in in_file_h5.root:
            actual_dut = int(node.name[-1:])
            if use_duts and actual_dut not in use_duts:
                continue
            logging.info('Calculate residuals for DUT %d', actual_dut)

            track_array = node[:]
            if max_chi2:
                track_array = track_array[track_array['track_chi2'] <= max_chi2]
            track_array = track_array[np.logical_and(track_array['column_dut_%d' % actual_dut] != 0., track_array['row_dut_%d' % actual_dut] != 0.)]  # take only tracks where actual dut has a hit, otherwise residual wrong
            hits, offset, slope = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0]))), np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
            intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane
            difference = intersection - hits

            # Calculate in um
            difference[:, 0] *= pixel_size[0]
            difference[:, 1] *= pixel_size[1]

            for i in range(2):  # col / row
                pixel_dim = pixel_size[i]
                hist, edges = np.histogram(difference[:, i], range=(-4. * pixel_dim, 4. * pixel_dim), bins=200)
                fit_ok = False
                try:
                    coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), 0., pixel_dim])
                    fit_ok = True
                except:
                    fit_ok = False

                plot_utils.plot_residuals(pixel_dim, i, actual_dut, edges, hist, fit_ok, coeff, gauss, difference, var_matrix, output_fig)

    if output_fig:
        output_fig.close()


def calculate_efficiency(tracks_file, output_pdf, z_positions, dim_x, dim_y, pixel_size, minimum_track_density, use_duts=None, max_chi2=None, cut_distance=500, max_distance=500, col_range=None, row_range=None):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    z_positions : iterable
        z_positions of all devices relative to DUT0
    dim_x, dim_y : integer
        front end dimensions of device
    pixel_size : iterable
        pixel dimensions
    minimum_track_density : int
        minimum track density required to consider bin for efficiency calculation
    use_duts : iterable
        the duts to calculate efficiency for. If None all duts are used
    max_chi2 : int
        only use track with a chi2 <= max_chi2
    cut_distance : int
        use only distances (between DUT hit and track hit) smaller than cut_distance
    max_distance : int
        defines binnig of distance values
    col_range, row_range : iterable
        column / row value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation)
    '''
    logging.info('Calculate efficiency')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(tracks_file, mode='r') as in_file_h5:
            for index, node in enumerate(in_file_h5.root):
                actual_dut = int(node.name[-1:])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Calculate efficiency for DUT %d', actual_dut)
                track_array = node[:]

                # Cut in Chi 2 of the track fit
                if max_chi2:
                    track_array = track_array[track_array['track_chi2'] <= max_chi2]

                # Take hits of actual DUT and track projection on actual DUT plane
                hits, offset, slope = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0]))), np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane

                # Select hits from column row range (e.g. to supress edge pixels)
                col_range = [col_range, ] if not isinstance(col_range, list) else col_range
                row_range = [row_range, ] if not isinstance(row_range, list) else row_range
                if len(col_range) == 1:
                    index = 0
                if len(row_range) == 1:
                    index = 0
                if None not in col_range:
                    selection = np.logical_and(intersection[:, 0] >= col_range[index][0], intersection[:, 0] <= col_range[index][1])
                    hits, intersection = hits[selection], intersection[selection]
                else:
                    col_range = [(0, dim_x)]
                if None not in row_range:
                    selection = np.logical_and(intersection[:, 1] >= row_range[index][0], intersection[:, 1] <= row_range[index][1])
                    hits, intersection = hits[selection], intersection[selection]
                else:
                    row_range = [(0, dim_y)]

                # Calculate distance between track hit and DUT hit
                scale = np.square(np.array((pixel_size[0], pixel_size[1], 0)))  # regard pixel size for calculating distances
                distance = np.sqrt(np.dot(np.square(intersection - hits), scale))  # array with distances between DUT hit and track hit for each event. Values in um

                col_row_distance = np.column_stack((hits[:, 0], hits[:, 1], distance))
                distance_array = np.histogramdd(col_row_distance, bins=(dim_x, dim_y, max_distance), range=[[1.5, dim_x + 0.5], [1.5, dim_y + 0.5], [0, max_distance]])[0]
                hit_hist, _, _ = np.histogram2d(hits[:, 0], hits[:, 1], bins=(dim_x, dim_y), range=[[1.5, dim_x + 0.5], [1.5, dim_y + 0.5]])

                # Calculate distances between hit and intersection
                distance_mean_array = np.average(distance_array, axis=2, weights=range(0, max_distance)) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_mean_array = np.ma.masked_invalid(distance_mean_array)
                distance_max_array = np.amax(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_min_array = np.amin(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_max_array = np.ma.masked_invalid(distance_max_array)
                distance_min_array = np.ma.masked_invalid(distance_min_array)

                # Calculate efficiency
                if cut_distance:  # select intersections where hit is distance
                    intersection_valid_hit = intersection[np.logical_and(np.logical_and(hits[:, 0] != 0, hits[:, 1] != 0), distance < cut_distance)]
                else:
                    intersection_valid_hit = intersection[np.logical_and(hits[:, 0] != 0, hits[:, 1] != 0)]

                plot_utils.efficiency_plots(distance_min_array, distance_max_array, actual_dut, intersection, minimum_track_density, intersection_valid_hit, hit_hist, distance_mean_array, dim_x, dim_y, cut_distance, output_fig)

                track_density, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(dim_x, dim_y), range=[[1.5, dim_x + 0.5], [1.5, dim_y + 0.5]])
                track_density_with_DUT_hit, _, _ = np.histogram2d(intersection_valid_hit[:, 0], intersection_valid_hit[:, 1], bins=(dim_x, dim_y), range=[[1.5, dim_x + 0.5], [1.5, dim_y + 0.5]])
                efficiency = np.zeros_like(track_density_with_DUT_hit)
                efficiency[track_density != 0] = track_density_with_DUT_hit[track_density != 0].astype(np.float) / track_density[track_density != 0].astype(np.float) * 100.
                efficiency = np.ma.array(efficiency, mask=track_density < minimum_track_density)

                logging.info('Efficiency =  %1.4f', np.ma.mean(efficiency))


# Helper functions that are not ment to be called during analysis

def _find_tracks_loop(tracklets, n_duts, column_sigma, row_sigma, pixel_ratio):
    ''' Complex loop to resort the tracklets array inplace to form track candidates. Each track candidate
    is given a quality identifier. Not ment to be called stand alone.
    Optimizations included to make it easily compile with numba in the future. Can be called from
    several real threads if they work on different areas of the array'''

    actual_event_number = tracklets[0]['event_number']
    n_tracks = tracklets.shape[0]
    # Numba does not understand python scopes, define all used variables here
    n_actual_tracks = 0
    track_index, actual_hit_track_index = 0, 0  # track index of table and first track index of actual event
    column, row = 0., 0.
    actual_track_column, actual_track_row = 0., 0.
    column_distance, row_distance = 0., 0.
    hit_distance = 0.
    best_hit_distance = 0.

    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=n_tracks, term_width=80)
    progress_bar.start()

    def set_track_quality(tracklets, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma):
        # Set track quality of actual DUT from closest DUT hit; if hit is within 2 or 5 sigma range; quality 0 already set
        column, row = tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index]
        if row != 0:  # row = 0: not hit
            column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
            if column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # high quality track hits
                actual_track['track_quality'] |= (65793 << dut_index)
            elif column_distance < 5 * actual_column_sigma and row_distance < 5 * actual_row_sigma:  # low quality track hits
                actual_track['track_quality'] |= (257 << dut_index)
        else:
            actual_track['track_quality'] &= (~ (65793 << dut_index))

    def swap_hits(tracklets, track_index, dut_index, column, row, charge):
        tmp_column, tmp_row, tmp_charge = tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index], tracklets[track_index]['charge_dut_%d' % dut_index]
        tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index], tracklets[track_index]['charge_dut_%d' % dut_index] = column, row, charge
        tracklets[hit_index]['column_dut_%d' % dut_index], tracklets[hit_index]['row_dut_%d' % dut_index], tracklets[hit_index]['charge_dut_%d' % dut_index] = tmp_column, tmp_row, tmp_charge

    for track_index, actual_track in enumerate(tracklets):  # loop over all possible tracks
        progress_bar.update(track_index)

        # Set variables for new event
        if actual_track['event_number'] != actual_event_number:
            actual_event_number = actual_track['event_number']
            for i in range(n_actual_tracks):  # Set number of tracks of previous event
                tracklets[track_index - 1 - i]['n_tracks'] = n_actual_tracks
            n_actual_tracks = 0
            actual_hit_track_index = track_index

        n_actual_tracks += 1
        first_hit_set = False

        for dut_index in xrange(n_duts):  # loop over all DUTs in the actual track

            actual_column_sigma, actual_row_sigma = column_sigma[dut_index], row_sigma[dut_index]

            if not first_hit_set and actual_track['row_dut_%d' % dut_index] != 0:  # search for first DUT that registered a hit (row != 0)
                actual_track_column, actual_track_row = actual_track['column_dut_%d' % dut_index], actual_track['row_dut_%d' % dut_index]
                first_hit_set = True
                actual_track['track_quality'] |= (65793 << dut_index)  # first track hit has best quality by definition
            else:  # Find best (closest) DUT hit
                close_hit_found = False
                for hit_index in xrange(actual_hit_track_index, tracklets.shape[0]):  # loop over all not sorted hits of actual DUT
                    if tracklets[hit_index]['event_number'] != actual_event_number:
                        break
                    column, row, charge, quality = tracklets[hit_index]['column_dut_%d' % dut_index], tracklets[hit_index]['row_dut_%d' % dut_index], tracklets[hit_index]['charge_dut_%d' % dut_index], tracklets[hit_index]['track_quality']
                    column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
                    hit_distance = sqrt((column_distance * pixel_ratio) * (column_distance * pixel_ratio) + row_distance * row_distance)

                    if row != 0:  # Track hit found
                        actual_track['track_quality'] |= (1 << dut_index)  # track quality 0 for DUT dut_index (in first byte one bit set)
                        quality |= (1 << dut_index)

                    if row != 0 and not close_hit_found and column_distance < 5. * actual_column_sigma and row_distance < 5. * actual_row_sigma:  # good track hit (5 sigma search region)
                        if tracklets[hit_index]['track_quality'] & (65793 << dut_index) == (65793 << dut_index):  # Check if hit is already a close hit, then do not move
                            set_track_quality(tracklets, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)
                            continue
                        if tracklets[hit_index]['track_quality'] & (257 << dut_index) == (257 << dut_index):  # Check if old hit is closer, then do not move
                            column_distance_old, row_distance_old = abs(column - tracklets[hit_index]['column_dut_0']), abs(row - tracklets[hit_index]['row_dut_0'])
                            hit_distance_old = sqrt((column_distance_old * pixel_ratio) * (column_distance_old * pixel_ratio) + row_distance_old * row_distance_old)
                            if hit_distance > hit_distance_old:  # Only take hit if it fits better to actual track
                                set_track_quality(tracklets, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)
                                continue
                        swap_hits(tracklets, track_index, dut_index, column, row, charge)
                        best_hit_distance = hit_distance
                        close_hit_found = True
                    elif row != 0 and close_hit_found and hit_distance < best_hit_distance:  # found better track hit
                        swap_hits(tracklets, track_index, dut_index, column, row, charge)
                        best_hit_distance = hit_distance

                    set_track_quality(tracklets, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)

        # Set number of tracks of last event
        for i in range(n_actual_tracks):
            tracklets[track_index - i]['n_tracks'] = n_actual_tracks

    progress_bar.finish()
    return tracklets


def _fit_tracks_loop(track_hits, pixel_dimension):
    ''' Do 3d line fit and calculate chi2 for each fit. '''
    def line_fit_3d(data, scale):
        datamean = data.mean(axis=0)
        offset, slope = datamean, np.linalg.svd(data - datamean)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
        intersections = offset + slope / slope[2] * (data.T[2][:, np.newaxis] - offset[2])  # fitted line and dut plane intersections (here: points)
        chi2 = np.sum(np.dot(np.square(data - intersections), scale), dtype=np.uint32)  # chi2 of the fit in pixel dimension units
        return datamean, slope, chi2

    slope = np.zeros((track_hits.shape[0], 3, ))
    offset = np.zeros((track_hits.shape[0], 3, ))
    chi2 = np.zeros((track_hits.shape[0], ))

    scale = np.square(np.array((pixel_dimension[0], pixel_dimension[-1], 0)))

    for index, actual_hits in enumerate(track_hits):  # loop over selected track candidate hits and fit
        try:
            offset[index], slope[index], chi2[index] = line_fit_3d(actual_hits, scale)
        except np.linalg.linalg.LinAlgError:
            chi2[index] = 1e9

    return offset, slope, chi2


def _function_wrapper_find_tracks_loop(args):  # needed for multiprocessing call with arguments
    return _find_tracks_loop(*args)


def _function_wrapper_fit_tracks_loop(args):  # needed for multiprocessing call with arguments
    return _fit_tracks_loop(*args)


if __name__ == "__main__":
    print 'Please check examples how to use the code'
