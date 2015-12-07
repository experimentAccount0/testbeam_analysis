from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib import colors, cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tables as tb
from math import sqrt, ceil
import logging
import re
import numpy as np

from testbeam_analysis import analysis_utils


def plot_noisy_pixel(occupancy, noisy_pixels, threshold, filename):
    # Plot noisy pixel
    plot_range = (occupancy.shape[0], occupancy.shape[1])
    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    print 'occupancy', np.median(occupancy) + np.std(occupancy), np.std(occupancy[occupancy < 10]), np.mean(occupancy), occupancy
    analysis_utils.create_2d_pixel_hist(fig, ax, occupancy.T, plot_range, title='Pixel map (%d hot pixel)' % noisy_pixels[0].shape[0], z_min=0, z_max=np.std(occupancy[occupancy < 10]) * threshold)
    fig.tight_layout()
    fig.savefig(filename)


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
                plt.ylim(1e-1, plt.ylim()[1])
                output_fig.savefig()


def plot_correlation_fit(x, y, coeff, var_matrix, xlabel, title, output_fig):
    def gauss(x, *p):
        A, mu, sigma, offset = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + offset
    plt.clf()
    gauss_fit_legend_entry = 'Gauss fit: \nA=$%.1f\pm %.1f$\nmu=$%.1f\pm% .1f$\nsigma=$%.1f\pm %.1f$' % (coeff[0], np.absolute(var_matrix[0][0] ** 0.5), coeff[1], np.absolute(var_matrix[1][1] ** 0.5), coeff[2], np.absolute(var_matrix[2][2] ** 0.5))
    plt.bar(x, y, label='data')
    x_fit = np.arange(np.amin(x), np.amax(x), 0.1)
    y_fit = gauss(x_fit, *coeff)
    plt.plot(x_fit, y_fit, '-', label=gauss_fit_legend_entry)
    plt.legend(loc=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('#')
    plt.grid()
    output_fig.savefig()


def plot_alignments(data, selected_data, pixel_length, mean_fitted, fit_fn, mean_error_fitted, offset, result, node_index, i, title):
    plt.clf()
    plt.title(title + ', Fit %d' % i)
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], mean_fitted[selected_data], 'o-', label='Data prefit')
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], fit_fn(pixel_length * np.arange(data.shape[0]))[selected_data], '-', label='Prefit')
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], mean_error_fitted[selected_data] * 1000., 'o-', label='Error x 1000')
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], offset[selected_data] * 10., 'o-', label='Offset x 10')

    plt.ylim((np.min(offset[selected_data]), np.amax(mean_fitted[selected_data])))
    plt.xlim((np.min(np.arange(data.shape[0])[selected_data]), pixel_length * data.shape[0]))
    plt.xlabel('DUT%d' % result[node_index]['dut_x'])
    plt.ylabel('DUT0')
    plt.legend(loc=0)
    plt.show()


def plot_alignment_fit(data, selected_data, pixel_length, mean_fitted, fit_fn, fit, pcov, chi2, mean_error_fitted, result, node_index, i, title, output_fig):
    plt.clf()
    plt.errorbar(pixel_length * np.arange(data.shape[0])[selected_data], mean_fitted[selected_data], yerr=mean_error_fitted[selected_data], fmt='.')
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], mean_error_fitted[selected_data] * 1000., 'o-', label='Error x 1000')
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], (fit_fn(pixel_length * np.arange(data.shape[0])[selected_data]) - mean_fitted[selected_data]) * 10., 'o-', label='Offset x 10')
    fit_legend_entry = 'fit: c0+c1x+c2x^2\nc0=$%1.1e\pm%1.1e$\nc1=$%1.1e\pm%1.1e$\nc2=$%1.1e\pm%1.1e$' % (fit[0], np.absolute(pcov[0][0]) ** 0.5, fit[1], np.absolute(pcov[1][1]) ** 0.5, fit[2], np.absolute(pcov[2][2]) ** 0.5)
    plt.plot(pixel_length * np.arange(data.shape[0]), fit_fn(pixel_length * np.arange(data.shape[0])), '-', label=fit_legend_entry)
    plt.plot(pixel_length * np.arange(data.shape[0])[selected_data], chi2[selected_data] / 1.e7)
    plt.legend(loc=0)
    plt.title(title)
    plt.xlabel('DUT %s' % result[node_index]['dut_x'])
    plt.ylabel('DUT %s' % result[node_index]['dut_y'])
    plt.xlim((0, np.amax(pixel_length * np.arange(data.shape[0]))))
    plt.grid()
    output_fig.savefig()


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


def plot_hit_alignment(title, difference, particles, ref_dut_column, table_column, actual_median, actual_mean, output_fig, bins=100):
    plt.clf()
    plt.hist(difference, bins=bins, range=(-np.amax(particles[:][ref_dut_column]) / 1., np.amax(particles[:][ref_dut_column]) / 1.))
    try:
        plt.yscale('log')
    except ValueError:
        pass
    plt.xlabel('%s - %s' % (ref_dut_column, table_column))
    plt.ylabel('#')
    plt.title(title)
    plt.grid()
    plt.plot([actual_median, actual_median], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Median %1.1f' % actual_median)
    plt.plot([actual_mean, actual_mean], [0, plt.ylim()[1]], '-', linewidth=2.0, label='Mean %1.1f' % actual_mean)
    plt.legend(loc=0)
    output_fig.savefig()


def plot_hit_alignment_2(in_file_h5, combine_n_hits, median, mean, correlation, alignment, output_fig):
    plt.clf()
    plt.xlabel('Hits')
    plt.ylabel('Offset')
    plt.grid()
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), median, linewidth=2.0, label='Median')
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), mean, linewidth=2.0, label='Mean')
    plt.plot(range(0, in_file_h5.root.Tracklets.shape[0], combine_n_hits), correlation, linewidth=2.0, label='Alignment')
    plt.legend(loc=0)
    output_fig.savefig()


def plot_z(z, dut_z_col, dut_z_row, dut_z_col_pos_errors, dut_z_row_pos_errors, dut_index, output_fig):
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


def plot_events(track_file, z_positions, event_range, dut=None, max_chi2=None, output_pdf=None):
    '''Plots the tracks (or track candidates) of the events in the given event range.

    Parameters
    ----------
    track_file : pytables file with tracks
    z_positions : iterable
    event_range : iterable:
        (start event number, stop event number(
    dut : int
        Take data from this DUT
    max_chi2 : int
        Plot only converged fits (cut on chi2)
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
        if max_chi2:
            tracks = tracks[tracks['track_chi2'] <= max_chi2]
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for track in tracks:
            x, y, z = [], [], []
            for dut_index in range(0, n_duts):
                if track['row_dut_%d' % dut_index] != 0:  # No hit has row = 0
                    x.append(track['column_dut_%d' % dut_index] * 1.e-3)  # in mm
                    y.append(track['row_dut_%d' % dut_index] * 1.e-3)  # in mm
                    z.append(z_positions[dut_index] * 1.e-3)  # in mm
            if fitted_tracks:
                offset = np.array((track['offset_0'], track['offset_1'], track['offset_2']))
                slope = np.array((track['slope_0'], track['slope_1'], track['slope_2']))
                linepts = offset * 1.e-3 + slope * 1.e-3 * np.mgrid[-100000:100000:2000j][:, np.newaxis]

            n_hits = bin(track['track_quality'] & 0xFF).count('1')
            n_very_good_hits = bin(track['track_quality'] & 0xFF0000).count('1')

            if n_hits > 2:  # only plot tracks with more than 2 hits
                if fitted_tracks:
                    ax.plot(x, y, z, '.' if n_hits == n_very_good_hits else 'o')
                    ax.plot3D(*linepts.T)
                else:
                    ax.plot(x, y, z, '.-' if n_hits == n_very_good_hits else '.--')

#         ax.set_xlim(0, 20)
#         ax.set_ylim(0, 20)
        ax.set_zlim(z_positions[0] * 1.e-3, z_positions[-1] * 1.e-3)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
        plt.title('%d tracks of %d events' % (tracks.shape[0], np.unique(tracks['event_number']).shape[0]))
        if output_pdf is not None:
            output_fig.savefig()
        else:
            plt.show()

    if output_fig:
        output_fig.close()


def plot_track_chi2(chi2s, fit_dut, output_fig):
    # Plot track chi2 and angular distribution
    plt.clf()
    plot_range = (0, 40000)
    plt.hist(chi2s, bins=200, range=plot_range)
    plt.grid()
    plt.xlim(plot_range)
    plt.xlabel('Track Chi2 [um*um]')
    plt.ylabel('#')
    plt.title('Track Chi2 for DUT %d tracks' % fit_dut)
    output_fig.savefig()


def plot_residuals(i, actual_dut, edges, hist, fit_ok, coeff, gauss, difference, var_matrix, output_fig):
    def get_rms_from_histogram(counts, bin_positions):
        values = []
        for index, one_bin in enumerate(counts):
            for _ in range(one_bin):
                values.append(bin_positions[index])
        return np.std(values)
    for plot_log in [False, True]:  # plot with log y or not
        plt.clf()
        plot_range = (-5 * get_rms_from_histogram(hist, edges), 5. * get_rms_from_histogram(hist, edges))
        plt.xlim(plot_range)
        plt.grid()
        plt.title('Residuals for DUT %d' % actual_dut)
        plt.xlabel('Residual Column [um]' if i == 0 else 'Residual Row [um]')
        plt.ylabel('#')

        if plot_log:
            plt.ylim(1, int(ceil(np.amax(hist) / 10.0)) * 100)

        plt.bar(edges[:-1], hist, width=(edges[1] - edges[0]), log=plot_log)
        if fit_ok:
            plt.plot([coeff[1], coeff[1]], [0, plt.ylim()[1]], color='red')
            plt.plot([np.median(difference[:, i]), np.median(difference[:, i])], [0, plt.ylim()[1]], '-', label='Median: $%.1f\pm %.1f$' % (np.median(difference[:, i]), 1.253 * np.std(difference[:, i]) / float(sqrt(difference[:, i].shape[0]))), color='green', linewidth=2)
            plt.plot([np.mean(difference[:, i]), np.mean(difference[:, i])], [0, plt.ylim()[1]], '-', label='Mean: $%.1f\pm %.1f$' % (np.mean(difference[:, i]), 1.253 * np.std(difference[:, i]) / float(sqrt(difference[:, i].shape[0]))), color='red', linewidth=2)
            gauss_fit_legend_entry = 'Gauss fit: \nA=$%.1f\pm %.1f$\nmu=$%.1f\pm %.1f$\nsigma=$%.1f\pm %.1f$' % (coeff[0], np.absolute(var_matrix[0][0] ** 0.5), coeff[1], np.absolute(var_matrix[1][1] ** 0.5), coeff[2], np.absolute(var_matrix[2][2] ** 0.5))
            plt.plot(np.arange(np.amin(edges[:-1]), np.amax(edges[:-1]), 0.1), gauss(np.arange(np.amin(edges[:-1]), np.amax(edges[:-1]), 0.1), *coeff), 'r--', label=gauss_fit_legend_entry, linewidth=2)
            plt.legend(loc=0)
        if output_fig is not None:
            output_fig.savefig()
        else:
            plt.show()


def plot_track_density(tracks_file, output_pdf, z_positions, dim_x, dim_y, pixel_size, mask_zero=True, use_duts=None, max_chi2=None):
    '''Takes the tracks and calculates the track density projected on selected DUTs.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    z_positions : iterable
        Iterable with z-positions of all DUTs
    dim_x, dim_y : integer
        front end dimensions of device (number of pixels)
    pixel_size : iterable
        pixel size (x, y) for every plane
    mask_zero : bool
        Mask heatmap entries = 0 for plotting
    use_duts : iterable
        the duts to plot track density for. If None all duts are used
    max_chi2 : int
        only use tracks with a chi2 <= max_chi2
    '''
    logging.info('Plot track density')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(tracks_file, mode='r') as in_file_h5:
            plot_ref_dut = False
            dimensions = []

            for index, node in enumerate(in_file_h5.root):
                # Bins define (virtual) pixel size for histogramming
                bin_x, bin_y = dim_x, dim_y

                # Calculate dimensions in um for every plane
                dimensions.append((dim_x * pixel_size[index][0], dim_y * pixel_size[index][1]))

                plot_range = (dimensions[index][0], dimensions[index][1])

                actual_dut = int(node.name[-1:])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Plot track density for DUT %d', actual_dut)

                track_array = node[:]

                # If set, select only converged fits
                if max_chi2:
                    track_array = track_array[track_array['track_chi2'] <= max_chi2]

                if plot_ref_dut:  # Plot first and last device
                    heatmap_ref_hits, _, _ = np.histogram2d(track_array['column_dut_0'], track_array['row_dut_0'], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                    if mask_zero:
                        heatmap_ref_hits = np.ma.array(heatmap_ref_hits, mask=(heatmap_ref_hits == 0))

                    # Get number of hits in DUT0
                    n_ref_hits = np.count_nonzero(heatmap_ref_hits)

                    fig = Figure()
                    fig.patch.set_facecolor('white')
                    ax = fig.add_subplot(111)
                    analysis_utils.create_2d_pixel_hist(fig, ax, heatmap_ref_hits.T, plot_range, title='Hit density for DUT 0 (%d Hits)' % n_ref_hits, x_axis_title="column [um]", y_axis_title="row [um]")
                    fig.tight_layout()
                    output_fig.savefig(fig)

                    plot_ref_dut = False

                offset, slope = np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane

                heatmap, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                heatmap_hits, _, _ = np.histogram2d(track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])

                # For better readability allow masking of entries that are zero
                if mask_zero:
                    heatmap = np.ma.array(heatmap, mask=(heatmap == 0))
                    heatmap_hits = np.ma.array(heatmap_hits, mask=(heatmap_hits == 0))

                # Get number of hits / tracks
                n_hits_heatmap = np.count_nonzero(heatmap)
                n_hits_heatmap_hits = np.count_nonzero(heatmap_hits)

                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                analysis_utils.create_2d_pixel_hist(fig, ax, heatmap.T, plot_range, title='Track density for DUT %d tracks (%d Tracks)' % (actual_dut, n_hits_heatmap), x_axis_title="column [um]", y_axis_title="row [um]")
                fig.tight_layout()
                output_fig.savefig(fig)

                fig = Figure()
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111)
                analysis_utils.create_2d_pixel_hist(fig, ax, heatmap_hits.T, plot_range, title='Hit density for DUT %d (%d Hits)' % (actual_dut, n_hits_heatmap_hits), x_axis_title="column [um]", y_axis_title="row [um]")
                fig.tight_layout()
                output_fig.savefig(fig)


def plot_charge_distribution(trackcandidates_file, output_pdf, dim_x, dim_y, pixel_size, mask_zero=True, use_duts=None):
    '''Takes the data and plots the charge distribution for selected DUTs.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    dim_x, dim_y : integer
        front end dimensions of device (number of pixels)
    pixel_size : iterable
        pixel size (x, y) for every plane
    mask_zero : bool
        Mask heatmap entries = 0 for plotting
    use_duts : iterable
        the duts to plot track density for. If None all duts are used
    '''
    logging.info('Plot charge distribution')
    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(trackcandidates_file, mode='r') as in_file_h5:
            dimensions = []
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'charge' in table_column:
                    actual_dut = int(table_column[-1:])

                    # Bins define (virtual) pixel size for histogramming
                    # TODO: allow different sensor sizes
                    n_bin_x, n_bin_y = dim_x, dim_y

                    # Calculate dimensions in um for every plane
                    index = actual_dut
                    dimensions.append((dim_x * pixel_size[index][0], dim_y * pixel_size[index][1]))
                    plot_range = (dimensions[index][0], dimensions[index][1])

                    if use_duts and actual_dut not in use_duts:
                        continue
                    logging.info('Plot charge distribution for DUT %d', actual_dut)

                    track_array = in_file_h5.root.TrackCandidates[:]

                    n_bins_charge = int(np.amax(track_array['charge_dut_%d' % actual_dut]))

                    x_y_charge = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], track_array['charge_dut_%d' % actual_dut]))
                    hit_hist, _, _ = np.histogram2d(track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], bins=(n_bin_x, n_bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                    charge_distribution = np.histogramdd(x_y_charge, bins=(n_bin_x, n_bin_y, n_bins_charge), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5], [0, n_bins_charge]])[0]

                    charge_density = np.average(charge_distribution, axis=2, weights=range(0, n_bins_charge)) * sum(range(0, n_bins_charge)) / hit_hist.astype(float)
                    charge_density = np.ma.masked_invalid(charge_density)

                    fig = Figure()
                    fig.patch.set_facecolor('white')
                    ax = fig.add_subplot(111)
                    analysis_utils.create_2d_pixel_hist(fig, ax, charge_density.T, plot_range, title='Charge density for DUT %d' % actual_dut, x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=int(np.ma.average(charge_density) * 1.5))
                    fig.tight_layout()
                    output_fig.savefig(fig)


def efficiency_plots(distance_min_array, distance_max_array, distance_mean_array, hit_hist, track_density, track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, plot_range, cut_distance, output_fig, mask_zero=True):
    # get number of entries for every histogram
    n_hits_distance_min_array = distance_min_array.count()
    n_hits_distance_max_array = distance_max_array.count()
    n_hits_distance_mean_array = distance_mean_array.count()
    n_hits_hit_hist = np.count_nonzero(hit_hist)
    n_tracks_track_density = np.count_nonzero(track_density)
    n_tracks_track_density_with_DUT_hit = np.count_nonzero(track_density_with_DUT_hit)
    n_hits_efficiency = np.count_nonzero(efficiency)

    # for better readability allow masking of entries that are zero
    if mask_zero:
        hit_hist = np.ma.array(hit_hist, mask=(hit_hist == 0))
        track_density = np.ma.array(track_density, mask=(track_density == 0))
        track_density_with_DUT_hit = np.ma.array(track_density_with_DUT_hit, mask=(track_density_with_DUT_hit == 0))

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, distance_min_array.T, plot_range, title='Minimal distance for DUT %d (%d Hits)' % (actual_dut, n_hits_distance_min_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=125000)
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, distance_max_array.T, plot_range, title='Maximal distance for DUT %d (%d Hits)' % (actual_dut, n_hits_distance_max_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=125000)
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, distance_mean_array.T, plot_range, title='Weighted distance for DUT %d (%d Hits)' % (actual_dut, n_hits_distance_mean_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=cut_distance)
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, hit_hist.T, plot_range, title='Hit density for DUT %d (%d Hits)' % (actual_dut, n_hits_hit_hist), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, track_density.T, plot_range, title='Track density for DUT %d (%d Tracks)' % (actual_dut, n_tracks_track_density), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, track_density_with_DUT_hit.T, plot_range, title='Density of tracks with DUT hit for DUT %d (%d Tracks)' % (actual_dut, n_tracks_track_density_with_DUT_hit), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_fig.savefig(fig)

    fig = Figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    analysis_utils.create_2d_pixel_hist(fig, ax, efficiency.T, plot_range, title='Efficiency for DUT %d (%d Entries)' % (actual_dut, n_hits_efficiency), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0., z_max=100.)
    fig.tight_layout()
    output_fig.savefig(fig)

    plt.clf()
    plt.grid()
    plt.title('Efficiency per pixel')
    plt.xlabel('Efficiency [%]')
    plt.ylabel('#')
    plt.yscale('log')
    plt.title('Efficiency for DUT %d' % actual_dut)
    plt.xlim([-0.5, 101.5])
    plt.hist(efficiency.ravel(), bins=100, range=(1, 100))
    output_fig.savefig()
