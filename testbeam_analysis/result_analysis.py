''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''

import logging
import tables as tb
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

from testbeam_analysis import plot_utils


def calculate_residuals(tracks_file, z_positions, use_duts=None, max_chi2=None, output_pdf=None):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    tracks_file : string
        File name with the tracks table
    z_position : iterable
        The positions of the devices in z in cm
    use_duts : iterable
        The duts to calculate residuals for. If None all duts in the tracks_file are used
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
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    output_fig = PdfPages(output_pdf) if output_pdf else None

    residuals = []

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

            for i in range(2):  # col / row
                mean, rms = np.mean(difference[:, i]), np.std(difference[:, i])
                hist, edges = np.histogram(difference[:, i], range=(mean - 5. * rms, mean + 5. * rms), bins=1000)
                fit_ok = False
                try:
                    coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
                    fit_ok = True
                except:
                    fit_ok = False

                if output_pdf is not False:
                    plot_utils.plot_residuals(i, actual_dut, edges, hist, fit_ok, coeff, gauss, difference, var_matrix, output_fig=output_fig)
                residuals.append(np.abs(coeff[2]))

    if output_fig:
        output_fig.close()

    return residuals


def calculate_efficiency(tracks_file, output_pdf, z_positions, bin_size, minimum_track_density, sensor_size=None, use_duts=None, max_chi2=None, cut_distance=500, max_distance=500, col_range=None, row_range=None):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.
    Parameters
    ----------
    tracks_file : string
        file name with the tracks table
    output_pdf : pdf file name object
    z_positions : iterable
        z_positions of all devices relative to DUT0
    bin_size : iterable
        sizes of bins (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes
    minimum_track_density : int
        minimum track density required to consider bin for efficiency calculation
    sensor_size : iterable
        size of the used sensor in um. Give one tuple (x, y) for every plane or list of tuples for different planes
    use_duts : iterable
        the DUTs to calculate efficiency for. If None all duts are used
    max_chi2 : int
        only use track with a chi2 <= max_chi2
    cut_distance : int
        use only distances (between DUT hit and track hit) smaller than cut_distance
    max_distance : int
        defines binnig of distance values
    col_range, row_range : iterable
        column / row value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation)
    '''

    logging.info('=== Calculate efficiency ===')
    with PdfPages(output_pdf) as output_fig:
        efficiencies = []
        with tb.open_file(tracks_file, mode='r') as in_file_h5:
            for index, node in enumerate(in_file_h5.root):
                actual_dut = int(node.name[-1:])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Calculate efficiency for DUT %d', actual_dut)
                track_array = node[:]

                # Get pixel and bin sizes for calculations and plotting
                # Allow different sensor sizes for every plane
                if not sensor_size:
                    dimensions = (np.amax(track_array['column_dut_%d' % actual_dut]), np.amax(track_array['row_dut_%d' % actual_dut]))
                else:
                    dimensions = [sensor_size, ] if not isinstance(sensor_size, list) else sensor_size
                    if len(dimensions) == 1:
                        dimensions = dimensions[0]
                    else:
                        dimensions = dimensions[index]

                # Allow different bin_sizes for every plane
                bin_size = [bin_size, ] if not isinstance(bin_size, list) else bin_size
                if len(bin_size) != 1:
                    n_bin_x = dimensions[0] / bin_size[index][0]
                    n_bin_y = dimensions[1] / bin_size[index][1]
                else:
                    n_bin_x = dimensions[0] / bin_size[0][0]
                    n_bin_y = dimensions[1] / bin_size[0][1]

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
                if None not in row_range:
                    selection = np.logical_and(intersection[:, 1] >= row_range[index][0], intersection[:, 1] <= row_range[index][1])
                    hits, intersection = hits[selection], intersection[selection]

                # Calculate distance between track hit and DUT hit
                scale = np.square(np.array((1, 1, 0)))  # regard pixel size for calculating distances
                distance = np.sqrt(np.dot(np.square(intersection - hits), scale))  # array with distances between DUT hit and track hit for each event. Values in um

                col_row_distance = np.column_stack((hits[:, 0], hits[:, 1], distance))
                distance_array = np.histogramdd(col_row_distance, bins=(n_bin_x, n_bin_y, max_distance), range=[[1.5, dimensions[0] + 0.5], [1.5, dimensions[1] + 0.5], [0, max_distance]])[0]
                hit_hist, _, _ = np.histogram2d(hits[:, 0], hits[:, 1], bins=(n_bin_x, n_bin_y), range=[[1.5, dimensions[0] + 0.5], [1.5, dimensions[1] + 0.5]])

                # Calculate distances between hit and intersection
                distance_mean_array = np.average(distance_array, axis=2, weights=range(0, max_distance)) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_mean_array = np.ma.masked_invalid(distance_mean_array)
                distance_max_array = np.amax(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_min_array = np.amin(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_max_array = np.ma.masked_invalid(distance_max_array)
                distance_min_array = np.ma.masked_invalid(distance_min_array)

                # Calculate efficiency
                if cut_distance:  # Select intersections where hit is in given distance around track intersection
                    intersection_valid_hit = intersection[np.logical_and(np.logical_and(hits[:, 0] != 0, hits[:, 1] != 0), distance < cut_distance)]
                else:
                    intersection_valid_hit = intersection[np.logical_and(hits[:, 0] != 0, hits[:, 1] != 0)]

                track_density, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(n_bin_x, n_bin_y), range=[[1.5, dimensions[0] + 0.5], [1.5, dimensions[1] + 0.5]])
                track_density_with_DUT_hit, _, _ = np.histogram2d(intersection_valid_hit[:, 0], intersection_valid_hit[:, 1], bins=(n_bin_x, n_bin_y), range=[[1.5, dimensions[0] + 0.5], [1.5, dimensions[1] + 0.5]])
                efficiency = np.zeros_like(track_density_with_DUT_hit)
                efficiency[track_density != 0] = track_density_with_DUT_hit[track_density != 0].astype(np.float) / track_density[track_density != 0].astype(np.float) * 100.
                efficiency = np.ma.array(efficiency, mask=track_density < minimum_track_density)

                plot_utils.efficiency_plots(distance_min_array, distance_max_array, distance_mean_array, hit_hist, track_density, track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, dimensions, cut_distance, output_fig)

                logging.info('Efficiency =  %1.4f', np.ma.mean(efficiency))
                efficiencies.append(np.ma.mean(efficiency))

    return efficiencies
