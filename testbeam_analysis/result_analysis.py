''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''
from __future__ import division

import logging
import re
import os.path

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from testbeam_analysis import plot_utils
from testbeam_analysis import geometry_utils


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))


def calculate_residuals_kalman(input_tracks_file, z_positions, use_duts=None, max_chi2=None, output_pdf=None, method="Interpolation", geometryFile=None):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    input_tracks_file : string
        File name with the tracks table
    z_position : iterable
        The positions of the devices in z in cm
    use_duts : iterable
        The duts to calculate residuals for. If None all duts in the input_tracks_file are used
    max_chi2 : int
        Use only converged fits (cut on chi2)
    output_pdf : pdf file name
        If None plots are printed to screen.
        If False no plots are created.
    Returns
    -------
    A list of residuals in column row. e.g.: [Col residual DUT 0, Row residual DUT 0, Col residual DUT 1, Row residual DUT 1, ...]
    '''
    logging.info('=== Calculate residuals ===')

    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))

    output_fig = PdfPages(output_pdf) if output_pdf else None

    residuals = []

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        translations, rotations = geometry_utils.recontruct_geometry_from_file(geometryFile)
        for node in in_file_h5.root:
            actual_dut = int(re.findall(r'\d+', node.name)[-1])
            if use_duts and actual_dut not in use_duts:
                continue
            logging.info('Calculate residuals for DUT %d', actual_dut)

            track_array = node[:]

            if max_chi2:
                track_array = track_array[track_array['track_chi2'] <= max_chi2]
            track_array = track_array[np.logical_and(track_array['column_dut_%d' % actual_dut] != 0., track_array['row_dut_%d' % actual_dut] != 0.)]  # take only tracks where actual dut has a hit, otherwise residual wrong
            if method == "Interpolation2":
                hits, offset, slope = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0]))), np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane
            elif method == "Kalman" or method == "Interpolation":
                hits, intersection = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0]))), np.column_stack((track_array['predicted_x%d' % actual_dut], track_array['predicted_y%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0])))

            tmpc = hits[:, 0] * rotations[actual_dut, 0, 0] + hits[:, 1] * rotations[actual_dut, 0, 1] + translations[actual_dut, 0]
            tmpr = hits[:, 0] * rotations[actual_dut, 1, 0] + hits[:, 1] * rotations[actual_dut, 1, 1] + translations[actual_dut, 1]
            hits[:, 0] = tmpc
            hits[:, 1] = tmpr

            difference = hits - intersection

            for i in range(2):  # col / row
                mean, rms = np.mean(difference[:, i]), np.std(difference[:, i])
                hist, edges = np.histogram(difference[:, i], range=(mean - 5.0 * rms, mean + 5.0 * rms), bins=1000)
                fit_ok = False
                try:
                    coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
                    fit_ok = True
                except:
                    fit_ok = False

                if output_pdf is not False:
                    plot_utils.plot_residuals(i, actual_dut, edges, hist, fit_ok, coeff, gauss, difference, var_matrix, output_fig=output_fig)
                residuals.append(np.abs(coeff[2]))

                for j in range(2):
                    _, xedges, yedges = np.histogram2d(hits[:, i], difference[:, j], bins=[100, 100], range=[[np.amin(hits[:, i]), np.amax(hits[:, i])], [-100, 100]])
                    plot_utils.plot_residuals_correlations(i, j, actual_dut, xedges, yedges, hits[:, i], difference[:, j], output_fig)
#                    s = analysis_utils.hist_2d_index(hits[:,i], difference[:,j], shape=(50,50))
                    # if j != i:
                    mean_fitted, selected_data, fit, pcov = calculate_correlation_fromplot(hits[:, i], difference[:, j], xedges, yedges, dofit=True)
                    plot_utils.plot_residuals_correlations_fit(i, j, actual_dut, xedges, yedges, mean_fitted, selected_data, fit, pcov, output_fig)

    if output_fig:
        output_fig.close()

    return residuals


def calculate_residuals(input_tracks_file, z_positions, dut_names=None, output_pdf_file=None, use_duts=None, max_chi2=None):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    input_tracks_file : string
        File name with the tracks table
    z_position : iterable
        The positions of the devices in z in cm
    use_duts : iterable
        The duts to calculate residuals for. If None all duts in the input_tracks_file are used
    max_chi2 : int
        Use only converged fits (cut on chi2)
    Returns
    -------
    A list of residuals in column row. e.g.: [Col residual DUT 0, Row residual DUT 0, Col residual DUT 1, Row residual DUT 1, ...]
    '''
    logging.info('=== Calculate residuals ===')

    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))

    output_pdf = PdfPages(output_pdf_file) if output_pdf_file else None

    residuals = []

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        for node in in_file_h5.root:
            actual_dut = int(re.findall(r'\d+', node.name)[-1])
            if use_duts and actual_dut not in use_duts:
                continue
            dut_name = dut_names[actual_dut] if dut_names else ("DUT " + str(actual_dut))
            logging.info('Calculate residuals for DUT %d', actual_dut)

            track_array = node[:]

            if max_chi2:
                track_array = track_array[track_array['track_chi2'] <= max_chi2]
            track_array = track_array[np.logical_and(track_array['column_dut_%d' % actual_dut] != 0., track_array['row_dut_%d' % actual_dut] != 0.)]  # take only tracks where actual dut has a hit, otherwise residual wrong
            hits = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], np.repeat(z_positions[actual_dut], track_array.shape[0])))
            offset = np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2']))
            slope = np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
            # intersection track with DUT plane
            intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])
            difference = intersection - hits
#             where = np.where(difference[:, 0] > 200.0)[0]
#             print where, len(where)
#             print intersection[where[0]], intersection.shape
#             print hits[where[0]], hits.shape

            for i in range(2):  # col / row
                mean, rms = np.mean(difference[:, i]), np.std(difference[:, i])
                hist, edges = np.histogram(difference[:, i], range=(mean - 5.0 * rms, mean + 5.0 * rms), bins=1000)
                fit_ok = False
                coeff, var_matrix = None, None
                try:
                    coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
                    fit_ok = True
                    residuals.append(np.abs(coeff[2]))
                except:
                    fit_ok = False
                    residuals.append(-1)

                plot_utils.plot_residuals(i, dut_name, edges, hist, fit_ok, coeff, gauss, difference, var_matrix, output_pdf=output_pdf)

    if output_pdf:
        output_pdf.close()

    return residuals


def calculate_correlation_fromplot(data1, data2, edges1, edges2, dofit=True):
    step = edges1[1] - edges1[0]
    nbins = len(edges1)
    resx = [[]] * len(edges1)
    mean_fitted = np.zeros(nbins)
    mean_error_fitted = np.zeros(nbins)

    for i, x in enumerate(data1):
        n = np.int((x - edges1[0]) / step)
        resx[n].append(data2[i])

    for n in range(nbins):
        if len(resx[n]) == 0:
            mean_fitted[n] = -1
            continue
        p0 = [np.amax(resx[n]), 0, 10]
        hist, edges = np.histogram(resx[n], range=(edges2[0], edges2[-1]), bins=len(edges2))
        ed = (edges[:-1] + edges[1:]) / 2.
        try:
            coeff, var_matrix = curve_fit(gauss, ed, hist, p0=p0)
            if var_matrix[1, 1] > 0.1:
                ''' > 0.01 for kalman'''
                ''' TOFIX: cut must be parameter!'''
                mean_fitted[n] = -1
                continue
            mean_fitted[n] = coeff[1]
            mean_error_fitted[n] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
            # sigma_fitted[index] = coeff[2]
        except RuntimeError:
            pass

    mean_fitted[~np.isfinite(mean_fitted)] = -1
    selected_data = np.where(np.logical_and(mean_fitted != -1, 1 > 0))[0]

    def f(x, c0, c1):
        return c0 + c1 * x

    if dofit:
        fit, pcov = curve_fit(f, edges1[selected_data], mean_fitted[selected_data])
    else:
        fit, pcov = None, None

    print("Linear fit:")
    print(fit)
    print(pcov)

    return mean_fitted, selected_data, fit, pcov


def calculate_efficiency(input_tracks_file, input_alignment_file, output_efficiency_file, z_positions, pixel_size, n_pixels, output_pdf_file=None, dut_names=None, bin_size=None, intersection_range=None, bin_range=None, charge_bins=None, minimum_tracks_per_bin=2, use_duts=None, max_chi2=None, cut_distance=500, max_distance=500, dut_masks=None):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.
    Parameters
    ----------
    input_tracks_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    z_positions : iterable
        z_positions of all devices relative to DUT0
    bin_size : iterable
        sizes of bins (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes
    minimum_tracks_per_bin : float
        minimum track density required to consider bin for efficiency calculation
    use_duts : iterable
        the DUTs to calculate efficiency for. If None all duts are used
    max_chi2 : int
        only use track with a chi2 <= max_chi2
    cut_distance : int
        use only distances (between DUT hit and track hit) smaller than cut_distance
    max_distance : int
        defines binnig of distance values
    '''

    # TODO:
    # bin_range  DUT col row range
    logging.info('=== Calculate efficiency ===')

    # calculate sensor size
    sensor_sizes = np.array(pixel_size) * n_pixels

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(output_efficiency_file)[0] + '.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(output_efficiency_file, 'w') as out_file_h5:
            with tb.open_file(input_alignment_file, mode='r') as in_alignment_file_h5:
                alignment = in_alignment_file_h5.root.Alignment[:]
                with tb.open_file(input_tracks_file, mode='r') as in_tracks_file_h5:
                    if use_duts is None:
                        use_duts = []
                        for node in in_tracks_file_h5.root:
                            use_duts.append(int(re.findall(r'\d+', node.name)[-1]))

                    if bin_size and len(use_duts) != len(bin_size):
                        raise ValueError('"bin_size" has not the length of "use_duts"')

                    if intersection_range and len(use_duts) != len(intersection_range):
                        raise ValueError('"intersection_range" has not the length of "use_duts"')

                    if charge_bins and len(use_duts) != len(charge_bins):
                        raise ValueError('"charge_bins" has not the length of "use_duts"')

                    for node in in_tracks_file_h5.root:
                        index = int(re.findall(r'\d+', node.name)[-1])
                        if use_duts and index not in use_duts:
                            continue
                        # index of use_duts
                        dut_index = np.where(np.array(use_duts) == index)[0][0]
                        dut_name = dut_names[index] if dut_names else ("DUT " + str(index))
                        logging.info('Calculating efficiency for DUT %d', index)
                        track_array = node[:]

                        # Allow different bin_sizes for every DUT plane
                        if bin_size and bin_size[dut_index]:
                            actual_bin_size_x = bin_size[dut_index][0]
                            actual_bin_size_y = bin_size[dut_index][1]
                        else:
                            actual_bin_size_x = pixel_size[index][0]
                            actual_bin_size_y = pixel_size[index][1]

                        sensor_size = sensor_sizes[index]
                        print "sensor_size", sensor_size

                        n_bin_x = sensor_size[0] / actual_bin_size_x
                        n_bin_y = sensor_size[1] / actual_bin_size_y
                        if not n_bin_x.is_integer() or not n_bin_y.is_integer():
                            raise ValueError("change bin_size")
                        n_bin_x = int(n_bin_x)
                        n_bin_y = int(n_bin_y)
                        # has to be even
                        print "bins", n_bin_x, n_bin_y

                        sensor_alignment = alignment[np.where(alignment["dut_x"] == index)]
                        if sensor_alignment.shape[0] != 0:
                            sensor_offset = [sensor_alignment[0]["c0"], sensor_alignment[1]["c0"]]
                            print "sensor_offset", sensor_offset
                            sensor_slope = [sensor_alignment[0]["c1"], sensor_alignment[1]["c1"]]
                            print "sensor_slope", sensor_slope
                        else:
                            sensor_offset = [1.0, 1.0]
                            print "sensor_offset", sensor_offset
                            sensor_slope = [0.0, 0.0]
                            print "sensor_slope", sensor_slope

                        sensor_range = [[sensor_offset[0], sensor_size[0] + sensor_offset[0]], [sensor_offset[1], sensor_size[1] + sensor_offset[1]]]
                        print "sensor_range", sensor_range

                        sensor_range_corr = [[sensor_offset[0], sensor_size[0] * sensor_slope[0] + sensor_offset[0]], [sensor_offset[1], sensor_size[1] * sensor_slope[1] + sensor_offset[1]]]
                        flip_column = False
                        flip_row = False
                        if sensor_slope[0] < 0.0:
                            flip_column = True
                            sensor_range[0][:] = [sensor_offset[0] - sensor_size[0], sensor_offset[0]]
                            sensor_range_corr[0][:] = sensor_range_corr[0][::-1]
                            print "correct column"
                        if sensor_slope[1] < 0.0:
                            flip_row = True
                            sensor_range[1][:] = [sensor_offset[1] - sensor_size[1], sensor_offset[1]]
                            sensor_range_corr[1][:] = sensor_range_corr[1][::-1]
                            print "correct row"
                        print "sensor_range", sensor_range
                        print "sensor_range_corr", sensor_range_corr

                        # cutting chi^2 of the track fit
                        if max_chi2 is not None:
                            track_array = track_array[track_array['track_chi2'] <= max_chi2]

                        # position of the cluster hits of the actual DUT
                        hits = np.column_stack((track_array['column_dut_%d' % index], track_array['row_dut_%d' % index], np.repeat(z_positions[index], track_array.shape[0]), track_array['charge_dut_%d' % index]))
                        # position of the track in the DUT plane
                        offset = np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2']))
                        slope = np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                        intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[index] - offset[:, 2, np.newaxis])
                        offset, slope = None, None

                        # select hits from column and row range
                        if intersection_range and intersection_range[dut_index]:
                            selection = np.logical_and(intersection[:, 0] >= intersection_range[dut_index][0][0], intersection[:, 0] <= intersection_range[dut_index][0][1])
                            hits = hits[selection]
                            intersection = intersection[selection]
                            selection = np.logical_and(intersection[:, 1] >= intersection_range[dut_index][1][0], intersection[:, 1] <= intersection_range[dut_index][1][1])
                            hits = hits[selection]
                            intersection = intersection[selection]

                        # calculate track density here, not masking 0, 0 pixels
                        tracks_per_bin, edge_x, edge_y = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)
                        tracks_per_bin = tracks_per_bin.astype(np.int32)
#                         tracks_per_bin = tracks_per_bin / (actual_bin_size_x * actual_bin_size_y)
                        print "max tracks per bin", np.max(tracks_per_bin)

                        # calculate distance between track hit and DUT cluster hit
                        # scale the size for calculation of the distance
                        scale = np.square(np.array((1, 1, 1)))
                        # array with distances between DUT hit and track hit for each event, in um
                        distance = np.sqrt(np.dot(np.square(intersection - hits[:, :3]), scale))

                        # intersection with hit
                        selection = np.logical_and(hits[:, 0] != 0.0, hits[:, 1] != 0.0)

                        if cut_distance is None:
                            hits_with_hit = hits[selection]
                            intersection_with_hit = intersection[selection]
                        else:
                            # select where hit is whithin given distance around track
                            new_selection = np.logical_and(selection, distance < cut_distance)
                            hits_with_hit = hits[new_selection]
                            intersection_with_hit = intersection[new_selection]
                        hits = hits[selection]

                        # efficiency
                        tracks_per_bin_with_hit, _, _ = np.histogram2d(intersection_with_hit[:, 0], intersection_with_hit[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)
                        tracks_per_bin_with_hit = tracks_per_bin_with_hit.astype(np.int32)
#                         tracks_per_bin_with_hit = tracks_per_bin_with_hit / (actual_bin_size_x * actual_bin_size_y)
                        efficiency = np.empty_like(tracks_per_bin_with_hit, dtype=np.float)
                        efficiency.fill(np.nan)
                        efficiency[tracks_per_bin != 0] = tracks_per_bin_with_hit[tracks_per_bin != 0].astype(np.float) / tracks_per_bin[tracks_per_bin != 0].astype(np.float) * 100.0
                        efficiency = np.ma.masked_invalid(efficiency)
                        efficiency = np.ma.masked_where(tracks_per_bin < minimum_tracks_per_bin, efficiency)

                        # calculate distances between hit and intersection
#                         col_row_distance = np.column_stack((hits[:, 0], hits[:, 1], distance))
                        # interesting effect
                        col_row_distance = np.column_stack((intersection[:, 0], intersection[:, 1], distance))
                        sensor_range_corr_with_distance = sensor_range_corr[:]
                        sensor_range_corr_with_distance.append([0, max_distance])
                        distance_array = np.histogramdd(col_row_distance, bins=(n_bin_x, n_bin_y, 100), range=sensor_range_corr_with_distance)[0]
                        hit_hist, _, _ = np.histogram2d(hits[:, 0], hits[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)
                        distance_mean_array = np.average(distance_array, axis=2, weights=range(0, 100)) * sum(range(0, 100)) / np.sum(distance_array, axis=2)
#                         distance_mean_array = np.average(distance_array, axis=2, weights=range(0, 100)) * sum(range(0, 100)) / hit_hist.astype(np.float)

                        distance_mean_array = np.ma.masked_invalid(distance_mean_array)
#                         distance_max_array = np.amax(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
#                         distance_min_array = np.amin(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
#                         distance_max_array = np.ma.masked_invalid(distance_max_array)
#                         distance_min_array = np.ma.masked_invalid(distance_min_array)

                        # calculate charge
                        mean_charge_array = None
                        if charge_bins and charge_bins[dut_index]:
#                             col_row_charge = np.column_stack((hits_with_hit[:, 0], hits_with_hit[:, 1], hits_with_hit[:, 3]))
                            col_row_charge = np.column_stack((intersection_with_hit[:, 0], intersection_with_hit[:, 1], hits_with_hit[:, 3]))
                            sensor_range_corr_with_charge = sensor_range_corr[:]
                            sensor_range_corr_with_charge.append([0, charge_bins[dut_index]])
                            charge_array = np.histogramdd(col_row_charge, bins=(n_bin_x, n_bin_y, charge_bins[dut_index]), range=sensor_range_corr_with_charge)[0]
                            mean_charge_array = np.average(charge_array, axis=2, weights=range(0, charge_bins[dut_index])) * sum(range(0, charge_bins[dut_index])) / np.sum(charge_array, axis=2)
                            mean_charge_array = np.ma.masked_invalid(mean_charge_array)

                            fig = Figure()
                            FigureCanvas(fig)
                            fig.patch.set_facecolor('white')
                            ax = fig.add_subplot(111)
                            ax.grid()
                            ax.set_title('Charge for %s' % (dut_name))
                            ax.set_xlabel('Charge')
                            ax.set_ylabel('#')
#                             ax.set_yscale('log')
                            ax.set_xlim(0, charge_bins[dut_index])
#                             print col_row_charge[:, 2].shape
#                             print col_row_charge[:, 2][col_row_charge[:, 2] == 0].shape
                            ax.hist(col_row_charge[:, 2], bins=charge_bins[dut_index], range=(0, charge_bins[dut_index]))  # Histogram not masked pixel efficiency
                            text = '$\mathrm{Mean:\ } %.2f$' % (np.mean(col_row_charge[:, 2]))
                            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                            ax.text(0.85, 0.9, text, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
                            output_pdf.savefig(fig)

                        print "bins with tracks", np.ma.count(efficiency), "of", efficiency.shape[0] * efficiency.shape[1]
                        print "col outside left / right", np.where(hits[:, 0] < sensor_range_corr[0][0])[0].shape[0], np.where(hits[:, 0] > sensor_range_corr[0][1])[0].shape[0]
                        print "row outside below / above", np.where(hits[:, 1] < sensor_range_corr[1][0])[0].shape[0], np.where(hits[:, 1] > sensor_range_corr[1][1])[0].shape[0]

                        if pixel_size:
                            aspect = pixel_size[index][1] / pixel_size[index][0]
                        else:
                            aspect = "auto"

                        plot_utils.efficiency_plots(distance_mean_array, hit_hist, tracks_per_bin, tracks_per_bin_with_hit, efficiency, dut_name, minimum_tracks_per_bin, mean_charge_array=mean_charge_array, plot_range=sensor_range_corr, n_pixels=n_pixels[index], cut_distance=cut_distance, output_fig=output_pdf, aspect=aspect, flip_column=flip_column, flip_row=flip_row)
                        logging.info('Efficiency =  %.2f', np.ma.mean(efficiency))

                        actual_dut_folder = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % index)
                        out_efficiency = out_file_h5.createCArray(actual_dut_folder, name='Efficiency', title='Efficiency per bin of DUT%d' % index, atom=tb.Atom.from_dtype(efficiency.dtype), shape=efficiency.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                        out_tracks_per_bin = out_file_h5.createCArray(actual_dut_folder, name='Tracks_per_bin', title='Tracks per bin of DUT%d' % index, atom=tb.Atom.from_dtype(tracks_per_bin.dtype), shape=tracks_per_bin.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                        # Store parameters used for efficiency calculation
                        out_efficiency.attrs.z_positions = z_positions
                        out_efficiency.attrs.bin_size = bin_size
                        out_efficiency.attrs.minimum_tracks_per_bin = minimum_tracks_per_bin
                        out_efficiency.attrs.sensor_size = sensor_size
                        out_efficiency.attrs.use_duts = use_duts
                        out_efficiency.attrs.max_chi2 = max_chi2
                        out_efficiency.attrs.cut_distance = cut_distance
                        out_efficiency.attrs.max_distance = max_distance
    #                         out_efficiency.attrs.col_range = col_range
    #                         out_efficiency.attrs.row_range = row_range
                        out_efficiency[:] = efficiency.T
                        out_tracks_per_bin[:] = tracks_per_bin.T

                        if dut_masks is not None:
                            dut_mask = dut_masks[dut_index]
                            # same DUT
                            if dut_mask is not None:
                                masked_pixels_row, masked_pixels_col = np.where(dut_mask > 0)
#                                 col_pos, col_retstep = np.linspace(sensor_range[0][0], sensor_range[0][1], n_pixels[index][0] + 1, endpoint=True, retstep=True)
#                                 col_pos = (col_pos[:-1] + col_pos[1:]) / 2
#                                 print "col_pos", col_pos, col_pos.shape, col_retstep
#                                 row_pos, row_retstep = np.linspace(sensor_range[1][0], sensor_range[1][1], n_pixels[index][1] + 1, endpoint=True, retstep=True)
#                                 row_pos = (row_pos[:-1] + row_pos[1:]) / 2
#                                 print "row_pos", row_pos, row_pos.shape, row_retstep
#                                 if flip_column:
#                                     col_pos = col_pos[::-1]
#                                 if flip_row:
#                                     row_pos = row_pos[::-1]
#                                 masked_positions = (row_pos[masked_pixels_row], col_pos[masked_pixels_col])
                                masked_positions = (masked_pixels_row + 1, masked_pixels_col + 1)
                                plot_utils.plot_efficiency_with_masked_pixels(efficiency=efficiency, n_pixels=n_pixels[index], dut_name=dut_name, output_fig=output_pdf, masked_positions=masked_positions, aspect=aspect, flip_column=flip_column, flip_row=flip_row)
