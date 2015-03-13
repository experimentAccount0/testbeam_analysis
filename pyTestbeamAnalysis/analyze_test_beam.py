"""This script does a full test beam analysis. As an input raw data files with a trigger number from one
run are expected. This script does in-RAM calculations on multiple cores in parallel. 12 Gb of free RAM and 8 cores are recommended.
The analysis flow is (also mentioned again in the __main__ section):
- Do for each DUT in parallel
  - Create a hit tables from the raw data
  - Align the hit table event number to the trigger number to be able to correlate hits in time
  - Cluster the hit table
- Create hit position correlations from the hit maps and store the arrays
- Plot the correlations as 2d heatmaps (optional)
- Take the correlation arrays and extract an offset/slope aligned to the first DUT
- Merge the cluster tables from all DUTs to one big cluster table and reference the cluster positions to the reference (DUT0) position
- Find tracks
- Align the DUT positions in z (optional)
- Fit tracks (very simple, fit straight line without hit correlstions taken into account)
- Plot event tracks (optional)
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
import numexpr as ne
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit, minimize_scalar

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib import colors, cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting
# from __future__ import print_function
# from numba import jit, numpy_support, types

from pyTestbeamAnalysis.hit_clusterizer import HitClusterizer
from pyTestbeamAnalysis.clusterizer import data_struct
from pyTestbeamAnalysis import analysis_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def remove_hot_pixels(data_file, threshold=2.):
    '''Std. analysis of a hit table. Clusters are created.

    Parameters
    ----------
    data_file : pytables file
    threshold : number
        The threshold when the pixel is removed given in sigma distance from the mean occupancy. 
    '''
    logging.info('Remove hot pixels in %s' % data_file)
    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_hot_pixel.h5', 'w') as out_file_h5:
            hits = input_file_h5.root.Hits[:]
            col, row = hits['column'], hits['row']
            occupancy = analysis_utils.hist_2d_index(col - 1, row - 1, shape=(np.amax(col), np.amax(row)))
            noisy_pixels = np.where(occupancy > np.mean(occupancy) + np.std(occupancy) * threshold)
            # Plot noisy pixel
            fig = Figure()
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            analysis_utils.create_2d_pixel_hist(fig, ax, occupancy.T, title='Pixel map (%d hot pixel)' % noisy_pixels[0].shape[0], z_min=0, z_max=np.std(occupancy) * threshold)
            fig.tight_layout()
            fig.savefig(data_file[:-3] + '_hot_pixel.pdf')
            logging.info('Remove %d hot pixels. Takes about %d seconds.' % (noisy_pixels[0].shape[0], noisy_pixels[0].shape[0] / 10))

            # Ugly solution to delete noisy pixels from the table with numexpression
            noisy_pixel_strings = ['((col==%d)&(row==%d))' % (noisy_pixel[0] + 1, noisy_pixel[1] + 1) for noisy_pixel in np.column_stack((noisy_pixels))]
            for one_slice in zip(*(iter(noisy_pixel_strings),) * (120 if len(noisy_pixel_strings) > 120 else len(noisy_pixel_strings))):
                col, row = hits['column'], hits['row']
                hits = hits[~ne.evaluate('|'.join(one_slice))]

            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=hits.dtype, title='Selected not noisy hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hit_table_out.append(hits)


def cluster_hits(data_file):
    '''Std. analysis of a hit table. Clusters are created.

    Parameters
    ----------
    data_file : pytables file
    output_file : pytables file
    '''

    logging.info('Cluster hits in %s' % data_file)

    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_cluster.h5', 'w') as output_file_h5:
            hits = input_file_h5.root.Hits[:]
            clusterizer = HitClusterizer(np.amax(hits['column']), np.amax(hits['row']))
            clusterizer.set_x_cluster_distance(1)  # cluster distance in columns
            clusterizer.set_y_cluster_distance(2)  # cluster distance in rows
            clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames
            cluster = np.zeros_like(hits, dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
            clusterizer.set_cluster_info_array(cluster)  # tell the array to be filled
            clusterizer.add_hits(hits)
            cluster = cluster[:clusterizer.get_n_clusters()]
            cluster_table_description = data_struct.ClusterInfoTable().columns.copy()
            cluster_table_out = output_file_h5.createTable(output_file_h5.root, name='Cluster', description=cluster_table_description, title='Clustered hits', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            cluster_table_out.append(cluster)


def plot_cluster_size(cluster_files, output_pdf):
    with PdfPages(output_pdf) as output_fig:
        for cluster_file in cluster_files:
            with tb.open_file(cluster_file, 'r') as input_file_h5:
                cluster = input_file_h5.root.Cluster[:]
                # Save cluster size histogram
                max_cluster_size = np.amax(cluster['size'])
                plt.clf()
                plt.bar(np.arange(max_cluster_size) + 0.6, analysis_utils.hist_1d_index(cluster['size'] - 1, shape=(max_cluster_size, )))
                plt.title('Cluster size of\n%s' % cluster_file)
                plt.xlabel('Cluster size')
                plt.ylabel('#')
                if max_cluster_size < 16:
                    plt.xticks(np.arange(0, max_cluster_size + 1, 1))
                plt.grid()
                plt.yscale('log')
                output_fig.savefig()


def correlate_hits(hit_files, alignment_file, fraction=1):
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
    '''
    logging.info('Correlate the position of %d DUTs' % len(hit_files))
    with tb.open_file(alignment_file, mode="w") as out_file_h5:
        for index, hit_file in enumerate(hit_files):
            with tb.open_file(hit_file, 'r') as in_file_h5:
                hit_table = in_file_h5.root.Hits[::fraction]
                if index == 0:
                    first_reference = pd.DataFrame({'event_number': hit_table[:]['event_number'], 'column_ref': hit_table[:]['column'], 'row_ref': hit_table[:]['row'], 'tot_ref': hit_table[:]['charge']})
                    n_col_reference, n_row_reference = np.amax(hit_table[:]['column']), np.amax(hit_table[:]['row'])
                else:
                    logging.info('Correlate detector %d with detector %d' % (index, 0))
                    dut = pd.DataFrame({'event_number': hit_table[:]['event_number'], 'column_dut': hit_table[:]['column'], 'row_dut': hit_table[:]['row'], 'tot_dut': hit_table[:]['charge']})
                    df = first_reference.merge(dut, how='left', on='event_number')
                    df.dropna(inplace=True)
                    n_col_dut, n_row_dut = np.amax(hit_table[:]['column']), np.amax(hit_table[:]['row'])
                    col_corr = analysis_utils.hist_2d_index(df['column_dut'] - 1, df['column_ref'] - 1, shape=(n_col_dut, n_col_reference))
                    row_corr = analysis_utils.hist_2d_index(df['row_dut'] - 1, df['row_ref'] - 1, shape=(n_row_dut, n_row_reference))
                    out = out_file_h5.createCArray(out_file_h5.root, name='CorrelationColumn_%d_0' % index, title='Column Correlation between DUT %d and %d' % (index, 0), atom=tb.Atom.from_dtype(col_corr.dtype), shape=col_corr.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    out_2 = out_file_h5.createCArray(out_file_h5.root, name='CorrelationRow_%d_0' % index, title='Row Correlation between DUT %d and %d' % (index, 0), atom=tb.Atom.from_dtype(row_corr.dtype), shape=row_corr.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    out.attrs.filenames = [str(hit_files[0]), str(hit_files[index])]
                    out_2.attrs.filenames = [str(hit_files[0]), str(hit_files[index])]
                    out[:] = col_corr
                    out_2[:] = row_corr


def align_hits(alignment_file, output_pdf, fit_offset_cut=(1.0, 1.0), fit_error_cut=(0.1, 0.1), show_plots=False):
    '''Takes the correlation histograms, determines usefull ranges with valid data, fits the correlations and stores the correlation parameters. With the
    correlation parameters one can calculate the hit position of each DUT in the master reference coordinate system. The fits are
    also plotted.

    Parameters
    ----------
    alignment_file : pytables file
        The input file with the correlation histograms and also the output file for correlation data.
    combine_bins : int
        Rebin the alignment histograms to get better statistics
    combine_bins : float
        Omit channels where the number of hits is < no_data_cut * mean channel hits
        Happens e.g. if the device is not fully illuminated
    fit_error_cut : float
        Omit channels where the fit has an error > fit_error_cut
        Happens e.g. if there is no clear correlation due to noise, isufficient statistics
    output_pdf : pdf file name object
    '''
    logging.info('Align hit coordinates')

    def gauss(x, *p):
        A, mu, sigma, offset = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(alignment_file, mode="r+") as in_file_h5:
            n_nodes = sum(1 for _ in enumerate(in_file_h5.root))  # Determine number of nodes, is there a better way?
            result = np.zeros(shape=(n_nodes,), dtype=[('dut_x', np.uint8), ('dut_y', np.uint8), ('c0', np.float), ('c0_error', np.float), ('c1', np.float), ('c1_error', np.float), ('c2', np.float), ('c2_error', np.float), ('c3', np.float), ('c3_error', np.float), ('sigma', np.float), ('sigma_error', np.float), ('description', np.str_, 40)])
            for node_index, node in enumerate(in_file_h5.root):
                try:
                    result[node_index]['dut_x'], result[node_index]['dut_y'] = int(re.search(r'\d+', node.name).group()), node.name[-1:]
                except AttributeError:
                    continue
                logging.info('Align %s' % node.name)

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
                        mean_error_fitted[index] = np.sqrt(np.diag(var_matrix))[1]
                        sigma_fitted[index] = coeff[2]
                        if index == data.shape[0] / 2:
                            plt.clf()
                            gauss_fit_legend_entry = 'Gaus fit: \nA=$%.1f\pm%.1f$\nmu=$%.1f\pm%.1f$\nsigma=$%.1f\pm%.1f$' % (coeff[0], np.absolute(var_matrix[0][0] ** 0.5), coeff[1], np.absolute(var_matrix[1][1] ** 0.5), coeff[2], np.absolute(var_matrix[2][2] ** 0.5))
                            plt.bar(x, data[index, :], label='data')
                            plt.plot(np.arange(np.amin(x), np.amax(x), 0.1), gauss(np.arange(np.amin(x), np.amax(x), 0.1), *coeff), '-', label=gauss_fit_legend_entry)
                            plt.legend(loc=0)
                            plt.title(node.title)
                            plt.xlabel('DUT 0 at DUT %s = %d' % (result[node_index]['dut_x'], index))
                            plt.ylabel('#')
                            plt.grid()
                            output_fig.savefig()
                    except RuntimeError:
                        pass

                mean_error_fitted = np.abs(mean_error_fitted)

                # Fit data with a straigth line 3 times to remove outliers
                selected_data = range(data.shape[0])
                for i in range(3):
                    f = lambda x, c0, c1: c0 + c1 * x
                    if not np.any(selected_data):
                        raise RuntimeError('The cuts are too tight, there is no point to fit. Release cuts and rerun alignment.')
                    offset_limit, error_limit = fit_offset_cut[0] if 'Col' in node.title else fit_offset_cut[1], fit_error_cut[0] if 'Col' in node.title else fit_error_cut[1]
                    fit, pcov = curve_fit(f, np.arange(data.shape[0])[selected_data], mean_fitted[selected_data])
                    fit_fn = np.poly1d(fit[::-1])
                    offset = fit_fn(np.arange(data.shape[0])) - mean_fitted
                    selected_data = np.where(np.logical_and(np.abs(offset) < offset_limit, mean_error_fitted < error_limit))
                    if show_plots:
                        plt.clf()
                        plt.title(node.title + ', Fit %d' % i)
                        plt.plot(np.arange(data.shape[0])[selected_data], mean_fitted[selected_data], 'o-', label='Data prefit')
                        plt.plot(np.arange(data.shape[0])[selected_data], fit_fn(np.arange(data.shape[0]))[selected_data], '-', label='Prefit')
                        plt.plot(np.arange(data.shape[0])[selected_data], mean_error_fitted[selected_data] * 1000., 'o-', label='Error x 1000')
                        plt.plot(np.arange(data.shape[0])[selected_data], offset[selected_data] * 10., 'o-', label='Offset x 10')

                        plt.ylim((np.min(offset[selected_data]), np.amax(mean_fitted[selected_data])))
                        plt.xlim((0, data.shape[0]))
                        plt.xlabel('DUT%d' % result[node_index]['dut_x'])
                        plt.ylabel('DUT0')
                        plt.legend(loc=0)
                        plt.show()

                # Refit with higher polynomial
                g = lambda x, c0, c1, c2, c3: c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3
                fit, pcov = curve_fit(g, np.arange(data.shape[0])[selected_data], mean_fitted[selected_data], sigma=mean_error_fitted[selected_data], absolute_sigma=True)
                fit_fn = np.poly1d(fit[::-1])

                # Calculate mean sigma (is somwhat a residual) and store the actual data in result array
                mean_sigma = np.mean(np.array(sigma_fitted)[selected_data])
                mean_sigma_error = np.std(np.array(sigma_fitted)[selected_data]) / np.sqrt(channel_indices[selected_data].shape[0])
                result[node_index]['c0'], result[node_index]['c0_error'] = fit[0], np.absolute(pcov[0][0]) ** 0.5
                result[node_index]['c1'], result[node_index]['c1_error'] = fit[1], np.absolute(pcov[1][1]) ** 0.5
                result[node_index]['c2'], result[node_index]['c2_error'] = fit[2], np.absolute(pcov[2][2]) ** 0.5
                result[node_index]['c3'], result[node_index]['c3_error'] = fit[3], np.absolute(pcov[3][3]) ** 0.5
                result[node_index]['sigma'], result[node_index]['sigma_error'] = mean_sigma, mean_sigma_error
                result[node_index]['description'] = node.title

                # Plot selected data with fit
                plt.clf()
                plt.errorbar(np.arange(data.shape[0])[selected_data], mean_fitted[selected_data], yerr=mean_error_fitted[selected_data], fmt='.')
                plt.plot(np.arange(data.shape[0])[selected_data], mean_error_fitted[selected_data] * 1000., 'o-', label='Error x 1000')
                plt.plot(np.arange(data.shape[0])[selected_data], (fit_fn(np.arange(data.shape[0])[selected_data]) - mean_fitted[selected_data]) * 10., 'o-', label='Offset x 10')
                fit_legend_entry = 'fit: c0+c1x+c2x^2+c3x^3\nc0=$%1.1e\pm%1.1e$\nc1=$%1.1e\pm%1.1e$\nc2=$%1.1e\pm%1.1e$\nc3=$%1.1e\pm%1.1e$' % (fit[0], np.absolute(pcov[0][0]) ** 0.5, fit[1], np.absolute(pcov[1][1]) ** 0.5, fit[2], np.absolute(pcov[2][2]) ** 0.5, fit[3], np.absolute(pcov[3][3]) ** 0.5)
                plt.plot(np.arange(data.shape[0]), fit_fn(np.arange(data.shape[0])), '-', label=fit_legend_entry)
                plt.plot(np.arange(data.shape[0])[selected_data], chi2[selected_data] / 1.e7)
                plt.legend(loc=0)
                plt.title(node.title)
                plt.xlabel('DUT %s' % result[node_index]['dut_y'])
                plt.ylabel('DUT %s' % result[node_index]['dut_x'])
                plt.xlim((0, np.amax(np.arange(data.shape[0]))))
                plt.grid()
                output_fig.savefig()

            try:
                result_table = in_file_h5.create_table(in_file_h5.root, name='Correlation', description=result.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                result_table.append(result)
            except tb.exceptions.NodeError:
                logging.warning('Correlation table exists already. Do not create new.')


def plot_correlations(alignment_file, output_pdf):
    '''Takes the correlation histograms and plots them

    Parameters
    ----------
    alignment_file : pytables file
        The input file with the correlation histograms and also the output file for correlation data.
    output_pdf : pdf file name object
    '''
    logging.info('Plotting Correlations')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(alignment_file, mode="r") as in_file_h5:
            for node in in_file_h5.root:
                try:
                    first, second = int(re.search(r'\d+', node.name).group()), node.name[-1:]
                except AttributeError:
                    continue
                data = node[:]
                plt.clf()
                cmap = cm.get_cmap('jet', 200)
                cmap.set_bad('w')
                norm = colors.LogNorm()
                z_max = np.amax(data)
                im = plt.imshow(data.T, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')
                divider = make_axes_locatable(plt.gca())
                plt.gca().invert_yaxis()
                plt.title(node.title)
                plt.xlabel('DUT %s' % first)
                plt.ylabel('DUT %s' % second)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax, ticks=np.linspace(start=0, stop=z_max, num=9, endpoint=True))
                output_fig.savefig()


def merge_cluster_data(cluster_files, alignment_file, tracklets_file):
    '''Takes the cluster from all cluster files and merges them into one big table onto the event number.
    Empty entries are signalled with charge = 0. The position is referenced from the correlation data to the first plane.
    Function uses easily several Gb of RAM. If memory errors occur buy a better PC or chunk this function.

    Parameters
    ----------
    cluster_files : list of pytables files
        Files with cluster data
    alignment_file : pytables files
        The file with the correlation data
    track_candidates_file : pytables files
    '''
    logging.info('Merge cluster to tracklets')
    with tb.open_file(alignment_file, mode="r") as in_file_h5:
        correlation = in_file_h5.root.Correlation[:]

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
                    c3 = np.array([0., 0.])
                else:
                    c0 = correlation[correlation['dut_x'] == index]['c0']
                    c1 = correlation[correlation['dut_x'] == index]['c1']
                    c2 = correlation[correlation['dut_x'] == index]['c2']
                    c3 = correlation[correlation['dut_x'] == index]['c3']

#                 print index, c0[0], c1[0], c2[0], c3[0], ',', c0[1], c1[1], c2[1], c3[1]
#                 print 'actual_mean_column, actual_mean_row',
#                 print actual_mean_column[0], actual_mean_row[0]
#                 print c1[0] * actual_mean_column[0] + c0[1], c1[1] * actual_mean_row[0] + c0[1]
#                 print 'corrected_mean_column, corrected_mean_row',
#                 print c3[0] * actual_mean_column[0] ** 3 + c2[0] * actual_mean_column[0] ** 2 + c1[0] * actual_mean_column[0] + c0[0],
#                 print c3[1] * actual_mean_row[0] ** 3 + c2[1] * actual_mean_row[0] ** 2 + c1[1] * actual_mean_row[0] + c0[1]
                tracklets_array['column_dut_%d' % index][selection] = c3[0] * actual_mean_column ** 3 + c2[0] * actual_mean_column ** 2 + c1[0] * actual_mean_column + c0[0]
                tracklets_array['row_dut_%d' % index][selection] = c3[1] * actual_mean_row ** 3 + c2[1] * actual_mean_row ** 2 + c1[1] * actual_mean_row + c0[1]
                tracklets_array['charge_dut_%d' % index][selection] = actual_cluster['charge'][selection]

#         np.nan_to_num(tracklets_array)
        tracklets_array['event_number'] = common_event_number
        tracklets_table.append(tracklets_array)


def optimize_hit_alignment(tracklets_files, use_fraction=0.1):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    use_fraction : float
        The fraction of hits to used for the alignment correction. For speed up. 1 means all hits are used
    '''
    logging.info('Optimize hit alignment')
    with tb.open_file(tracklets_files, mode="r+") as in_file_h5:
        particles = in_file_h5.root.Tracklets[:]
        for table_column in in_file_h5.root.Tracklets.dtype.names:
            if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                ref_dut_column = table_column[:-1] + '0'
                logging.info('Optimize alignment for % s', table_column)
                every_nth_hit = int(1. / use_fraction)
                particle_selection = particles[::every_nth_hit][np.logical_and(particles[::every_nth_hit][ref_dut_column] > 0, particles[::every_nth_hit][table_column] > 0)]  # only select events with hits in both DUTs
                difference = particle_selection[ref_dut_column] - particle_selection[table_column]
                selection = np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)  # select all hits from events with hits in both DUTs
                particles[table_column][selection] += np.median(difference)
        in_file_h5.removeNode(in_file_h5.root, 'Tracklets')
        corrected_tracklets_table = in_file_h5.create_table(in_file_h5.root, name='Tracklets', description=particles.dtype, title='Tracklets', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        corrected_tracklets_table.append(particles)


def check_hit_alignment(tracklets_files, output_pdf, combine_n_events=100000):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).

    Parameters
    ----------
    tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf : pdf file name object
    combine_n_events : int
        The number of events to combine for the hit position check
    '''
    logging.info('Align hit coordinates')
    with tb.open_file(tracklets_files, mode="r") as in_file_h5:
        with PdfPages(output_pdf) as output_fig:
            for table_column in in_file_h5.root.Tracklets.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    median, mean, std = [], [], []
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Check alignment for % s', table_column)
                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Tracklets.shape[0], term_width=80)
                    progress_bar.start()
                    for index in range(0, in_file_h5.root.Tracklets.shape[0], combine_n_events):
                        particles = in_file_h5.root.Tracklets[index:index + combine_n_events]
                        particles = particles[np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)]  # only select events with hits in both DUTs
                        plt.clf()
                        plt.hist(particles[:][ref_dut_column] - particles[:][table_column], bins=100, range=(-np.amax(particles[:][ref_dut_column]) / 10., np.amax(particles[:][ref_dut_column]) / 10.))
                        plt.xlabel('%s - %s' % (ref_dut_column, table_column))
                        plt.ylabel('#')
                        plt.title('Aligned position difference for events %d - %d' % (index, index + combine_n_events))
                        plt.grid()
                        difference = particles[:][ref_dut_column] - particles[:][table_column]
                        actual_median, actual_mean, actual_rms = np.median(difference), np.mean(difference), np.std(difference)
                        median.append(actual_median)
                        mean.append(actual_mean)
                        std.append(actual_rms)
                        plt.plot([actual_median, actual_median], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Median %1.1f' % actual_median)
                        plt.plot([actual_mean, actual_mean], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Mean %1.1f' % actual_mean)
                        plt.legend(loc=0)
                        output_fig.savefig()
                        progress_bar.update(index)
                    plt.clf()
                    plt.xlabel('Event')
                    plt.ylabel('Offset')
                    plt.grid()
                    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_events), median, linewidth=2.0, label='Median')
                    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_events), mean, linewidth=2.0, label='Mean')
                    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_events), std, linewidth=2.0, label='RMS')
                    plt.legend(loc=0)
                    output_fig.savefig()
                    progress_bar.finish()


def find_tracks(tracklets_file, alignment_file, track_candidates_file):
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

    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Correlation[:]
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
        arg = [(one_slice, correlations, n_duts, column_sigma, row_sigma) for one_slice in slices]  # FIXME: slices are not aligned at event numbers, up to n_slices * 2 tracks are found wrong
        results = pool.map(_function_wrapper_find_tracks_loop, arg)
        result = np.concatenate(results)

# _find_tracks_loop_compiled = jit((numpy_support.from_dtype(tracklets.dtype)[:], types.int32, types.float64, types.float64), nopython=True)(_find_tracks_loop)  # maybe in 1 year this will help, when numba works with structured arrays
#         _find_tracks_loop(tracklets, correlations, n_duts, column_sigma, row_sigma)
#         result = tracklets

        with tb.open_file(track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=in_file_h5.root.Tracklets.description, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            track_candidates.append(result)


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
                logging.info('Find best z-position for DUT %d' % dut_index)
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

                # Plot actual DUT data
                plt.clf()
                plt.plot([dut_z_col.x, dut_z_col.x], [0., 1.], "--", label="DUT%d, col, z=%1.4f" % (dut_index, dut_z_col.x))
                plt.plot([dut_z_row.x, dut_z_row.x], [0., 1.], "--", label="DUT%d, row, z=%1.4f" % (dut_index, dut_z_row.x))
                plt.plot(z, dut_z_col_pos_errors / np.amax(dut_z_col_pos_errors), "-", label="DUT%d, column" % dut_index)
                plt.plot(z, dut_z_row_pos_errors / np.amax(dut_z_row_pos_errors), "-", label="DUT%d, row" % dut_index)
                plt.grid()
                plt.legend(loc=1)
                plt.ylim((np.amin(dut_z_col_pos_errors / np.amax(dut_z_col_pos_errors)), 1.))
                plt.xlabel('Relative z-position')
                plt.ylabel('Mean squared offset [a.u.]')
                plt.gca().set_yscale('log')
                plt.gca().get_yaxis().set_ticks([])
                output_fig.savefig()

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
        logging.info('Absoulte reconstructed z-positions %s' % str(z_positions_rec_abs))
        if failing_duts:
            logging.warning('The reconstructed z postions is more than %1.1f cm off for DUTS %s' % (warn_at, str(failing_duts)))
        else:
            logging.info('Difference between measured and reconstructed z-positions %s' % str(z_differences))

    return z_positions_rec_abs if z_positions is not None else z_positions_rec


def event_display(track_file, z_positions, event_range, pixel_size=(250, 50), plot_lim=(2, 2), dut=None, output_pdf=None):
    '''Plots the tracks (or track candidates) of the events in the given event range.

    Parameters
    ----------
    track_file : pytables file with tracks
    z_positions : iterable
    event_range : iterable:
        (start event number, stop event number(
    pixel_size : iterable:
        (column size, row size) in um
    plot_lim : iterable:
        (column lim, row lim) in cm
    output_pdf : pdf file name
    '''

    output_fig = PdfPages(output_pdf) if output_pdf else None

    with tb.open_file(track_file, "r") as in_file_h5:
        fitted_tracks = False
        try:  # data has track candidates
            table = in_file_h5.root.TrackCandidates
        except tb.NoSuchNodeError:  # data has fitted tracks
            table = in_file_h5.getNode(in_file_h5.root, name='Tracks_DUT_%d' % dut)
            fitted_tracks = True

        n_duts = sum(['column' in col for col in table.dtype.names])
        array = table[:]
        tracks = analysis_utils.get_data_in_event_range(array, event_range[0], event_range[-1])
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for track in tracks:
            x, y, z = [], [], []
            for dut_index in range(0, n_duts):
                if track['row_dut_%d' % dut_index] != 0:  # No hit has row = 0
                    x.append(track['column_dut_%d' % dut_index] * pixel_size[0] * 1e-3)
                    y.append(track['row_dut_%d' % dut_index] * pixel_size[1] * 1e-3)
                    z.append(z_positions[dut_index])
            if fitted_tracks:
                scale = np.array((pixel_size[0] * 1e-3, pixel_size[1] * 1e-3, 1))
                offset = np.array((track['offset_0'], track['offset_1'], track['offset_2'])) * scale
                slope = np.array((track['slope_0'], track['slope_1'], track['slope_2'])) * scale
                linepts = offset + slope * np.mgrid[-100:100:2j][:, np.newaxis]

            n_hits = bin(track['track_quality'] & 0xFF).count('1')
            n_very_good_hits = bin(track['track_quality'] & 0xFF0000).count('1')

            if n_hits > 2:  # only plot tracks with more than 2 hits
                if fitted_tracks:
                    ax.plot(x, y, z, '.' if n_hits == n_very_good_hits else 'o')
                    ax.plot3D(*linepts.T)
                else:
                    ax.plot(x, y, z, '.-' if n_hits == n_very_good_hits else '.--')

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_zlim(z_positions[0], z_positions[-1])
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [cm]')
        plt.title('%d track of %d events' % (tracks.shape[0], np.unique(tracks['event_number']).shape[0]))
        if output_pdf is not None:
            output_fig.savefig()
        else:
            plt.show()

    if output_fig:
        output_fig.close()


def fit_tracks(track_candidates_file, tracks_file, z_positions, fit_duts=None, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], max_tracks=1, track_quality=1, pixel_size=(250, 50), output_pdf=None):
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
        the duts that are not taken in a fit. Needed to exclude bad planes from track fit.
    include_duts : iterable
        the relative dut positions of dut to use in the track fit. The position is relative to the actual dut the tracks are fitted for
        e.g. actual track fit dut = 2, include_duts = [-3, -2, -1, 1] means that duts 0, 1, 3 are used for the track fit
    output_pdf : pdf file name object
        if None plots are printed to screen
    max_tracks : int
        only events with tracks <= max tracks are taken
    pixel_size : iterable, (x dimensions, y dimension)
        the size in um of the pixels, needed for ch2 calculation
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
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
                n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                track_candidates = in_file_h5.root.TrackCandidates[:]
                fit_duts = fit_duts if fit_duts else range(n_duts)
                for fit_dut in fit_duts:  # loop over the duts to fit the tracks for
                    logging.info('Fit tracks for DUT %d' % fit_dut)

                    # Select track candidates
                    dut_selection = 0
                    for include_dut in include_duts:  # calculate mask to select DUT hits for fitting
                        if fit_dut + include_dut < 0 or (ignore_duts and fit_dut + include_dut in ignore_duts):
                            continue
                        if include_dut >= 0:
                            dut_selection |= ((1 << (n_duts - 1)) >> fit_dut) >> include_dut
                        else:
                            dut_selection |= ((1 << (n_duts - 1)) >> fit_dut) << abs(include_dut)

                    if bin(dut_selection).count("1") < 2:
                        logging.warning('Insufficient track hits to do fit (< 2). Omit DUT %d' % fit_dut)
                        continue
                    good_track_selection = np.logical_and((track_candidates['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8)), track_candidates['n_tracks'] <= max_tracks)
                    good_track_candidates = track_candidates[good_track_selection]

                    # Prepare track hits array to be fitted
                    n_fit_duts = n_duts - len(ignore_duts) - 1 if ignore_duts else n_duts - 1
                    n_fit_duts += 1 if ignore_duts and fit_dut in ignore_duts else 0
                    n_fit_duts += 1 if 0 in include_duts else 0
                    tmp_dut_index, n_tracks = 0, good_track_candidates['event_number'].shape[0]
                    track_hits = np.zeros((n_tracks, n_fit_duts, 3))
                    for dut_index in range(0, n_duts):
                        if (ignore_duts and dut_index in ignore_duts) or (0 not in include_duts and dut_index == fit_dut):
                            continue
                        xyz = np.column_stack((good_track_candidates['column_dut_%s' % dut_index], good_track_candidates['row_dut_%s' % dut_index], np.repeat(z_positions[dut_index], n_tracks)))
                        track_hits[:, tmp_dut_index, :] = xyz
                        tmp_dut_index += 1

                    # Split data and fit on all available cores
                    n_slices = cpu_count() - 1
                    slice_length = n_tracks / n_slices
                    slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
                    pool = Pool(n_slices)
                    arg = [(one_slice, pixel_size) for one_slice in slices]  # FIXME: slices are not aligned at event numbers, up to n_slices * 2 tracks are found wrong
                    results = pool.map(_function_wrapper_fit_tracks_loop, arg)
                    del track_hits

                    # Store results
                    offsets = np.concatenate([i[0] for i in results])  # merge offsets from all cores in results
                    slopes = np.concatenate([i[1] for i in results])  # merge slopes from all cores in results
                    chi2s = np.concatenate([i[2] for i in results])  # merge slopes from all cores in results
                    tracks_array = create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts)
                    tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    tracklets_table.append(tracks_array)

                    # Plot track chi2 and angular distribution
                    plt.clf()
                    plot_range = (0, 40000)
                    plt.hist(chi2s, bins=200, range=plot_range)
                    plt.grid()
                    plt.xlim(plot_range)
                    plt.xlabel('Track Chi2 [um*um]')
                    plt.ylabel('#')
                    plt.title('Track Chi2 for DUT %d tracks' % fit_dut)
                    plt.gca().set_yscale('log')
                    output_fig.savefig()


def calculate_residuals(tracks_file, z_positions, pixel_size=(50, 50), use_duts=None, max_chi2=None, track_quality=1, output_pdf=None):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    z_position : iterable
        the positions of the devices in z in cm
    use_duts : iterable
        the duts to calculate residuals for. If None all duts are used
    output_pdf : pdf file name
        if None plots are printed to screen
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
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
            logging.info('Calculate residuals for DUT %d' % actual_dut)

            track_array = node[:]
            if max_chi2:
                track_array = track_array[track_array['track_chi2'] <= max_chi2]
            track_array = track_array[track_array['charge_dut_%d' % actual_dut] != 0]  # take only tracks where actual dut has a hit, otherwise residual wrong
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
                for plot_log in [False, True]:  # plot with log y or not
                    plt.clf()
                    plot_range = (-i - 4.5 * pixel_dim, i + 4.5 * pixel_dim)
                    plt.xlim(plot_range)
                    plt.grid()
                    plt.title('Residuals for DUT %d' % actual_dut)
                    plt.xlabel('Residual Column [um]' if i == 0 else 'Residual Row [um]')
                    plt.ylabel('#')
                    plt.bar(edges[:-1], hist, width=(edges[1] - edges[0]), log=plot_log)
                    if fit_ok:
                        plt.plot([coeff[1], coeff[1]], [0, plt.ylim()[1]], color='red')
                        gauss_fit_legend_entry = 'gaus fit: \nA=$%.1f\pm%.1f$\nmu=$%.1f\pm%.1f$\nsigma=$%.1f\pm%.1f$' % (coeff[0], np.absolute(var_matrix[0][0] ** 0.5), coeff[1], np.absolute(var_matrix[1][1] ** 0.5), coeff[2], np.absolute(var_matrix[2][2] ** 0.5))
                        plt.plot(np.arange(np.amin(edges[:-1]), np.amax(edges[:-1]), 0.1), gauss(np.arange(np.amin(edges[:-1]), np.amax(edges[:-1]), 0.1), *coeff), 'r-', label=gauss_fit_legend_entry, linewidth=2)
                        plt.legend(loc=0)
                    if output_fig is not None:
                        output_fig.savefig()
                    else:
                        plt.show()

    if output_fig:
        output_fig.close()


def plot_track_density(tracks_file, output_pdf, z_positions, dim_x, dim_y, use_duts=None, max_chi2=None):
    '''Takes the tracks and calculates the track density projected on selected DUTs.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    use_duts : iterable
        the duts to plot track density for. If None all duts are used
    output_pdf : pdf file name object
    max_chi2 : int
        only use track with a chi2 <= max_chi2
    '''
    logging.info('Plot track density')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(tracks_file, mode='r') as in_file_h5:
            for node in in_file_h5.root:
                actual_dut = int(node.name[-1:])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Plot track density for DUT %d' % actual_dut)

                track_array = node[:]
                track_array = track_array[track_array['track_chi2'] != 1000000000] # use tracks with converged fit only

                offset, slope = np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane
                if max_chi2:
                    intersection = intersection[track_array['track_chi2'] <= max_chi2]
                
                heatmap, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(dim_x, dim_y), range=[[1, dim_x], [1, dim_y]])

                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                analysis_utils.create_2d_pixel_hist(fig, ax, heatmap.T, title='Track densitiy for DUT %d tracks' % actual_dut, x_axis_title="column", y_axis_title="row")
                fig.tight_layout()
                output_fig.savefig(fig)


def calculate_efficiency(tracks_file, output_pdf, z_positions, dim_x, dim_y, pixel_size, minimum_track_density, use_duts=None, max_chi2=None):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    minimum_track_density : int
        minimum track density required to consider bin for efficiency calculation
    use_duts : iterable
        the duts to calculate efficiency for. If None all duts are used
    output_pdf : pdf file name object
    max_chi2 : int
        only use track with a chi2 <= max_chi2
    '''
    logging.info('Calculate efficiency')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(tracks_file, mode='r') as in_file_h5:
            for node in in_file_h5.root:
                actual_dut = int(node.name[-1:])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Calculate efficiency for DUT %d' % actual_dut)

                track_array = node[:]
                track_array = track_array[track_array['track_chi2'] != 1000000000] # use tracks with converged fit only

                if max_chi2:
                    track_array = track_array[track_array['track_chi2'] <= max_chi2]
                hits, charge, offset, slope = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0]))), track_array['charge_dut_%d' % actual_dut], np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane

                # Calculate distance between track hit and DUT hit
                scale = np.square(np.array((pixel_size[0], pixel_size[1], 0)))
                distance = np.sqrt(np.dot(np.square(intersection - hits), scale))
                col_row_distance = np.column_stack((hits[:, 0], hits[:, 1], distance))
                distance_array = np.histogramdd(col_row_distance, bins=(dim_x, dim_y, 500), range=[[1, dim_x], [1, dim_y], [0, 500]])[0]
                hh, _, _ = np.histogram2d(hits[:, 0], hits[:, 1], bins=(dim_x, dim_y), range=[[1, dim_x], [1, dim_y]])
                distance_mean_array = np.average(distance_array, axis=2, weights=range(0, 500)) * sum(range(0, 500)) / hh.astype(np.float)
                distance_mean_array = np.ma.masked_invalid(distance_mean_array)
                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                analysis_utils.create_2d_pixel_hist(fig, ax, distance_mean_array.T, title='Distance for DUT %d' % actual_dut, x_axis_title="column", y_axis_title="row", z_min=0, z_max=150)
                fig.tight_layout()
                output_fig.savefig(fig)

                # Calculate efficiency
                track_density, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(dim_x, dim_y), range=[[1, dim_x], [1, dim_y]])
                track_density_with_DUT_hit, _, _ = np.histogram2d(intersection[charge != 0][:, 0], intersection[charge != 0][:, 1], bins=(dim_x, dim_y), range=[[1, dim_x], [1, dim_y]])
                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)

                efficiency = np.zeros_like(track_density_with_DUT_hit)
                efficiency[track_density != 0] = track_density_with_DUT_hit[track_density != 0].astype(np.float) / track_density[track_density != 0].astype(np.float) * 100.

                efficiency = np.ma.array(efficiency, mask=track_density < minimum_track_density)
                analysis_utils.create_2d_pixel_hist(fig, ax, efficiency.T, title='Efficiency for DUT %d' % actual_dut, x_axis_title="column", y_axis_title="row", z_min=50., z_max=100.)
                fig.tight_layout()
                output_fig.savefig(fig)
                plt.clf()
                plt.grid()
                plt.title('Efficiency per pixel')
                plt.xlabel('Efficiency [%]')
                plt.ylabel('#')
                plt.yscale('log')
                plt.title('Efficiency for DUT %d' % actual_dut)
                plt.hist(efficiency.ravel(), bins=100, range=(50, 104))
                output_fig.savefig()
                logging.info('Efficiency =  %1.4f' % np.ma.mean(efficiency[efficiency > 50].ravel()))


# Helper functions that are not ment to be called during analysis
def _find_tracks_loop(tracklets, correlations, n_duts, column_sigma, row_sigma):
    ''' Complex loop to resort the tracklets array inplace to form track candidates. Each track candidate
    is given a quality identifier. Not ment to be called stand alone.
    Optimizations included to make it easily compile with numba in the future. Can be called from
    several real threads if they work on different areas of the array'''

    actual_event_number = tracklets[0]['event_number']
    n_tracks = tracklets.shape[0]
    # Numba does not understand python scopes, define all used variables here
    n_actual_tracks = 0
    track_index = 0
    column, row = 0., 0.
    actual_track_column, actual_track_row = 0., 0.
    column_distance, row_distance = 0., 0.
    hit_distance = 0.
    tmp_column, tmp_row = 0., 0.
    best_hit_distance = 0.

    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=n_tracks, term_width=80)
    progress_bar.start()

    for track_index, actual_track in enumerate(tracklets):
        progress_bar.update(track_index)

        if actual_track['event_number'] != actual_event_number:
            actual_event_number = actual_track['event_number']
            for i in range(n_actual_tracks):
                tracklets[track_index - 1 - i]['n_tracks'] = n_actual_tracks
            n_actual_tracks = 0

        n_actual_tracks += 1
        first_hit_set = False

        for dut_index in xrange(n_duts):
            actual_column_sigma, actual_row_sigma = column_sigma[dut_index], row_sigma[dut_index]
            if not first_hit_set and actual_track['row_dut_%d' % dut_index] != 0:
                actual_track_column, actual_track_row = actual_track['column_dut_%d' % dut_index], actual_track['row_dut_%d' % dut_index]
                first_hit_set = True
                actual_track['track_quality'] |= (65793 << (n_duts - dut_index - 1))  # first track hit has best quality by definition
            else:
                # Find best (closest) DUT hit
                close_hit_found = False
                for hit_index in xrange(track_index, tracklets.shape[0]):  # loop over all not sorted hits of actual DUT
                    if tracklets[hit_index]['event_number'] != actual_event_number:
                        break
                    column, row = tracklets[hit_index]['column_dut_%d' % dut_index], tracklets[hit_index]['row_dut_%d' % dut_index]
                    column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
                    hit_distance = sqrt((column_distance * 5) * (column_distance * 5) + row_distance * row_distance)

                    if row != 0:  # Track hit found
                        actual_track['track_quality'] |= (1 << (n_duts - dut_index - 1))

                    if row != 0 and not close_hit_found and column_distance < 5 * actual_column_sigma and row_distance < 5 * actual_row_sigma:  # good track hit (5 sigma search region)
                        tmp_column, tmp_row = tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index]
                        tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index] = column, row
                        tracklets[hit_index]['column_dut_%d' % dut_index], tracklets[hit_index]['row_dut_%d' % dut_index] = tmp_column, tmp_row
                        best_hit_distance = hit_distance
                        close_hit_found = True
                    elif row != 0 and close_hit_found and hit_distance < best_hit_distance:  # found better track hit
                        tmp_column, tmp_row = tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index]
                        tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index] = column, row
                        tracklets[hit_index]['column_dut_%d' % dut_index], tracklets[hit_index]['row_dut_%d' % dut_index] = tmp_column, tmp_row
                        best_hit_distance = hit_distance
                    elif row == 0 and not close_hit_found:  # take no hit if no good hit is found
                        tmp_column, tmp_row = tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index]
                        tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index] = column, row
                        tracklets[hit_index]['column_dut_%d' % dut_index], tracklets[hit_index]['row_dut_%d' % dut_index] = tmp_column, tmp_row

                # Set track quality of actual DUT from closest DUT hit
                column, row = tracklets[track_index]['column_dut_%d' % dut_index], tracklets[track_index]['row_dut_%d' % dut_index]
                column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
                if column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # high quality track hits
                    actual_track['track_quality'] |= (65793 << (n_duts - dut_index - 1))
                elif column_distance < 5 * actual_column_sigma and row_distance < 5 * actual_row_sigma:  # low quality track hits
                    actual_track['track_quality'] |= (257 << (n_duts - dut_index - 1))
    else:
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
    print 'Check examples how to use the code'
