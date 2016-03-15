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
from scipy.optimize import curve_fit, minimize_scalar, leastsq, OptimizeWarning
from matplotlib.backends.backend_pdf import PdfPages

from multiprocessing import Pool, cpu_count
from math import sqrt
from math import asin
from math import acos

from testbeam_analysis import analysis_utils
from testbeam_analysis import plot_utils
from testbeam_analysis import geometry_utils


warnings.simplefilter("ignore", OptimizeWarning)  # Fit errors are handled internally, turn of warnings


def correlate_hits(input_hits_files, output_correlation_file, n_pixels, pixel_size=None, dut_names=None, output_pdf_file=None, chunk_size=4999999):
    '''Histograms the hit column (row) of two different devices on an event basis. If the hits are correlated a line should be seen.
    Permutations are not considered (not all hits of the first device are correlated with all hits of the second device).

    Parameters
    ----------
    input_hits_files : pytables file
        Input file with hit data.
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

    logging.info('=== Correlate the position of %d DUTs ===', len(input_hits_files))
    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        n_duts = len(input_hits_files)

        # Result arrays to be filled
        column_correlations = [None] * (n_duts - 1)
        row_correlations = [None] * (n_duts - 1)

        with tb.open_file(input_hits_files[0], mode='r') as in_file_h5:  # Open DUT0 hit file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Hits.shape[0], term_width=80)
            progress_bar.start()
            start_indices = [0] * (n_duts - 1)  # Store the loop indices for speed up
            for hits_dut_0, index in analysis_utils.data_aligned_at_events(in_file_h5.root.Hits, chunk_size=chunk_size):  # Loop over the hits of DUT0 in chunks
                actual_event_numbers = hits_dut_0[:]['event_number']
                # Calculate the common event number of each device with the reference device and correlate the hits of this events
                for dut_index, hit_file in enumerate(input_hits_files[1:], start=1):  # Loop over the other hit files
                    with tb.open_file(hit_file, mode='r') as actual_in_file_h5:  # Open other DUT hit file
                        for actual_dut_hits, start_indices[dut_index - 1] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Hits, start=start_indices[dut_index - 1], start_event_number=actual_event_numbers[0], stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size):  # Loop over the hits in the actual hit file in chunks
                            dut_0_hits, actual_dut_hits = analysis_utils.merge_on_event_number(hits_dut_0, actual_dut_hits)

                            if not np.any(column_correlations[dut_index - 1]):
                                column_correlations[dut_index - 1] = analysis_utils.hist_2d_index(actual_dut_hits['column'] - 1, dut_0_hits['column'] - 1, shape=(n_pixels[dut_index][0], n_pixels[0][0]))
                                row_correlations[dut_index - 1] = analysis_utils.hist_2d_index(actual_dut_hits['row'] - 1, dut_0_hits['row'] - 1, shape=(n_pixels[dut_index][1], n_pixels[0][1]))
                            else:
                                column_correlations[dut_index - 1] += analysis_utils.hist_2d_index(actual_dut_hits['column'] - 1, dut_0_hits['column'] - 1, shape=(n_pixels[dut_index][0], n_pixels[0][0]))
                                row_correlations[dut_index - 1] += analysis_utils.hist_2d_index(actual_dut_hits['row'] - 1, dut_0_hits['row'] - 1, shape=(n_pixels[dut_index][1], n_pixels[0][1]))

                progress_bar.update(index)

            # Store the correlation histograms
            for dut_index in range(n_duts - 1):
                out_col = out_file_h5.createCArray(out_file_h5.root, name='CorrelationColumn_%d_0' % (dut_index + 1), title='Column Correlation between DUT %d and %d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(column_correlations[dut_index].dtype), shape=column_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row = out_file_h5.createCArray(out_file_h5.root, name='CorrelationRow_%d_0' % (dut_index + 1), title='Row Correlation between DUT %d and %d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(row_correlations[dut_index].dtype), shape=row_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col.attrs.filenames = [str(input_hits_files[0]), str(input_hits_files[dut_index])]
                out_row.attrs.filenames = [str(input_hits_files[0]), str(input_hits_files[dut_index])]
                out_col[:] = column_correlations[dut_index]
                out_row[:] = row_correlations[dut_index]
            progress_bar.finish()

    plot_utils.plot_correlations(input_correlation_file=output_correlation_file, pixel_size=pixel_size, dut_names=dut_names)


def coarse_alignment(input_correlation_file, output_alignment_file, pixel_size, dut_names=None, output_pdf_file=None, non_interactive=False, iterations=3):
    '''Takes the correlation histograms, fits the correlations and stores the correlation parameters.
    The user can define cuts on the fit error and straight line offset in an interactive way.

    This is a coarse alignment that uses the hit correlation and corrects for translations between the planes and beam divergences.
    The alignment of the plane rotation needs the the fine alignment function.

    Parameters
    ----------
    input_correlation_file : pytbales file
        The input file with the correlation histograms.
    output_alignment_file : pytables file
        The output file for correlation data.
    pixel_size: iterable
        List of tuples with column and row pixel size. In um.
    dut_names: iterable
        List of names of the DUTs.
    output_pdf_file : string
        File name for the output plots.
    non_interactive : boolean
        Deactivate user interaction and apply cuts automatically
    iterations : number
        Only used in non interactive mode. Sets how often automatic cuts are applied.
    '''
    logging.info('=== Coarse aligning the DUTs ===')

    def gauss_offset(x, *p):
        A, mu, sigma, offset, slope = p
        return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0)) + offset + x * slope

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(output_alignment_file)[0] + '.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
            n_nodes = len(in_file_h5.list_nodes("/"))
            n_duts = int(n_nodes / 2) + 1
            result = np.zeros(shape=(n_nodes,), dtype=[('dut_x', np.uint8), ('dut_y', np.uint8), ('c0', np.float), ('c0_error', np.float), ('c1', np.float), ('c1_error', np.float), ('sigma', np.float), ('sigma_error', np.float)])
            for node in in_file_h5.root:
                indices = re.findall(r'\d+', node.name)
                dut_idx = int(indices[0])
                ref_idx = int(indices[1])
                if "column" in node.name.lower():
                    node_index = dut_idx - 1
                else:
                    node_index = dut_idx - 1 + int(n_nodes / 2)
                result[node_index]['dut_x'] = dut_idx
                result[node_index]['dut_y'] = ref_idx
                dut_name = dut_names[dut_idx] if dut_names else ("DUT " + str(dut_idx))
                ref_name = dut_names[ref_idx] if dut_names else ("DUT " + str(ref_idx))
                logging.info('Aligning %s', node.name)

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
                n_hits = np.array([-1.0 for _ in range(data.shape[0])])  # Number of hits per bin

                # Loop over all row/row or column/column slices and fit a double gaussian or gaussian + offset to the profile
                # Get values with highest correlation for alignment fit; do this with channel indices, later convert to um
                # Origin pixel cluster mean is 1.5 / 1.5, since hits start from 1, 1 not 0, 0

                x_hist_fit = np.arange(1.5, data.shape[1] + 1.5)  # x bin positions
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
                        n_hits[index] = data[index, :].sum()
                        fit_type = 1
                    except RuntimeError:
                        try:
                            p0 = [A_start[index], mu_start[index], 5.0, A_mean[index], 0.0]  # FIXME: hard coded start value
                            coeff, var_matrix = curve_fit(gauss_offset, x_hist_fit, data[index, :], p0=p0)
                            y_fit = gauss_offset(x_hist_fit, *coeff)
                            mean_fitted[index] = coeff[1]
                            mean_error_fitted[index] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
                            sigma_fitted[index] = np.abs(coeff[2])
                            n_hits[index] = data[index, :].sum()
                            fit_type = 2
                        except RuntimeError:
                            pass
                    finally:
                        # Create plot in the center of the mean data
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

                # Show the correlation fit/fit errors and offsets from straigt line
                # Let the user change the cuts (error limit, offset limit) and refit until result looks good
                refit = True
                selected_data = np.ones_like(mean_fitted, dtype=np.bool)
                x = np.arange(1.5, mean_fitted.shape[0] + 1.5) * pixel_length_dut
                actual_iteration = 0  # Refit counter for non interactive mode
                while(refit):
                    selected_data, fit, refit = plot_utils.plot_alignments(x=x,
                                                                           mean_fitted=mean_fitted,
                                                                           mean_error_fitted=mean_error_fitted,
                                                                           n_hits=n_hits,
                                                                           ref_name=ref_name,
                                                                           dut_name=dut_name,
                                                                           title="Correlation of %s: %s vs. %s" % ("columns" if "column" in node.name.lower() else "rows", dut_name, ref_name),
                                                                           non_interactive=non_interactive)
                    x = x[selected_data]
                    mean_fitted = mean_fitted[selected_data]
                    mean_error_fitted = mean_error_fitted[selected_data]
                    sigma_fitted = sigma_fitted[selected_data]
                    chi2 = chi2[selected_data]
                    n_hits = n_hits[selected_data]
                    # Stop in non interactive mode if the number of refits (iterations) is reached
                    if non_interactive:
                        actual_iteration += 1
                        if actual_iteration > iterations:
                            break

                # Linear fit, usually describes correlation very well
                # With low energy beam and / or beam with diverse agular distribution, the correlation will not be straight

                def line(x, c0, c1):
                    return c0 + c1 * x

                # Use results from straight line fit as start values for final fit
                re_fit, re_fit_pcov = curve_fit(line, x, mean_fitted, sigma=mean_error_fitted, absolute_sigma=True, p0=[fit[0], fit[1]])

                # Write fit results to array
                result[node_index]['c0'], result[node_index]['c0_error'] = re_fit[0], np.absolute(re_fit_pcov[0][0]) ** 0.5
                result[node_index]['c1'], result[node_index]['c1_error'] = re_fit[1], np.absolute(re_fit_pcov[1][1]) ** 0.5

                # Calculate mean sigma (is a residual when assuming straight tracks) and its error and store the actual data in result array
                # This error is needed for track finding
                mean_sigma = pixel_length_ref * np.mean(np.array(sigma_fitted))
                mean_sigma_error = pixel_length_ref * np.std(np.array(sigma_fitted)) / np.sqrt(np.array(sigma_fitted).shape[0])

                result[node_index]['sigma'], result[node_index]['sigma_error'] = mean_sigma, mean_sigma_error

                # Plot selected data with fit
                fit_fn = np.poly1d(re_fit[::-1])
                plot_utils.plot_alignment_fit(x=x, mean_fitted=mean_fitted, fit_fn=fit_fn, fit=re_fit, pcov=re_fit_pcov, chi2=chi2, mean_error_fitted=mean_error_fitted, dut_name=dut_name, ref_name=ref_name, title="Correlation of %s: %s vs. %s" % ("columns" if "column" in node.name.lower() else "rows", ref_name, dut_name), output_pdf=output_pdf)

            with tb.open_file(output_alignment_file, mode="w") as out_file_h5:
                try:
                    result_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', description=result.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    result_table.append(result)
                except tb.exceptions.NodeError:
                    logging.warning('Correlation table exists already. Do not create new.')

                # Create transilation / rotation table that can be can be overwritten later in the fine alignment step; initial values define no translation and no rotation
                description = [('DUT', np.int)]
                for index in range(3):  # Translation has 3 dimensions
                    description.append(('translation_%d' % index, np.float))
                for i in range(3):  # Rotation matrix of the DUT
                    for j in range(3):
                        description.append(('rotation_%d_%d' % (i, j), np.float))

                trans_rot_parameters = np.zeros((n_duts,), dtype=description)

                # Rotation matrix without effect has 1s in the diagonal
                trans_rot_parameters[:]['rotation_0_0'] = np.ones((n_duts,))
                trans_rot_parameters[:]['rotation_1_1'] = np.ones((n_duts,))
                trans_rot_parameters[:]['rotation_2_2'] = np.ones((n_duts,))

                try:
                    geometry_table = out_file_h5.create_table(out_file_h5.root, name='Geometry', title='File containing the fine alignment geometry parameters', description=np.zeros((1,), dtype=trans_rot_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    geometry_table.append(trans_rot_parameters)
                except tb.exceptions.NodeError:
                    logging.warning('Correlation table exists already. Do not create new.')


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

def fit_plot_tracks_slopes (slopes, actual_dut, output_fig):
    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    for i in range(2):  # col / row
        mean, rms = np.mean(slopes[:, i]), np.std(slopes[:, i])
        hist, edges = np.histogram(slopes[:, i], range=(mean - 5. * rms, mean + 5. * rms), bins=1000)
        fit_ok = False
        coeff, var_matrix = None, None
        try:
            coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
            fit_ok = True
        except:
            fit_ok = False

        plot_utils.plot_track_slope(i, actual_dut, edges, hist, fit_ok, coeff, gauss, slopes, var_matrix, output_fig=output_fig)


def fit_tracks_align(track_candidates_file, z_positions, fit_dut, geometry, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], track_quality=0, max_tracks=None, use_correlated=False, chunk_size=1000000):
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
    geometry: dictrionary for geometry parameters
    fit_dut : integer
        the dut to fit tracks for.
    ignore_duts : iterable
        the duts that are not taken in a fit. Needed to exclude bad planes from track fit. Also included Duts are ignored!
    include_duts : iterable
        the relative dut positions of dut to use in the track fit. The position is relative to the actual dut the tracks are fitted for
        e.g. actual track fit dut = 2, include_duts = [-3, -2, -1, 1] means that duts 0, 1, 3 are used for the track fit
    max_tracks : int, None
        only events with tracks <= max tracks are taken
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    pixel_size : iterable, (x dimensions, y dimension)
        the size in um of the pixels, needed for chi2 calculation
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''

    logging.info('=== Fit tracks ===')

    def create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts):
        # Define description
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

        # Define structure of track_array
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

    nplanes = len(z_positions)
    translations = np.zeros((nplanes, 3))
    rotations = np.zeros((nplanes, 3, 3))
    for index in range(3):
        translations[:, index] = geometry['translation_%d' % index]
    for i in range(3):
        for j in range(3):
            rotations[:, i, j] = geometry['rotation_%d_%d' % (i, j)]

    with tb.open_file(track_candidates_file, mode='r') as in_file_h5:
        n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
        logging.info('Fit tracks for DUT %d', fit_dut)
        #tracklets_table = None
        offsets, slopes, dut_hits = [], [], []
        n_chunk = 0  # TOFIT this is a bypass...
        for track_candidates_chunk, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates, chunk_size=chunk_size):
            # Select track candidates
            dut_selection = 0  # DUTs to be used in the fit
            quality_mask = 0  # Masks DUTs to check track quality for
            for include_dut in include_duts:  # Calculate mask to select DUT hits for fitting
                if fit_dut + include_dut < 0 or ((ignore_duts and fit_dut + include_dut in ignore_duts) or fit_dut + include_dut >= n_duts):
                    continue
                if include_dut >= 0:
                    dut_selection |= ((1 << fit_dut) << include_dut)
                else:
                    dut_selection |= ((1 << fit_dut) >> abs(include_dut))

                quality_mask = dut_selection | (1 << fit_dut)  # Include the DUT where the track is fitted for in quality check

            if bin(dut_selection).count("1") < 2:
                logging.warning('Insufficient track hits to do fit (< 2). Omit DUT %d', fit_dut)
                continue

            # Select tracks based on given track_quality
            good_track_selection = (track_candidates_chunk['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8))
            if max_tracks:  # Option to neglect events with too many hits
                good_track_selection = np.logical_and(good_track_selection, track_candidates_chunk['n_tracks'] <= max_tracks)

            logging.info('Lost %d tracks due to track quality cuts, %d percent ', good_track_selection.shape[0] - np.count_nonzero(good_track_selection), (1. - float(np.count_nonzero(good_track_selection) / float(good_track_selection.shape[0]))) * 100.)

            if use_correlated:  # Reduce track selection to correlated DUTs only
                good_track_selection &= (track_candidates_chunk['track_quality'] & (quality_mask << 24) == (quality_mask << 24))
                logging.info('Lost due to correlated cuts %d', good_track_selection.shape[0] - np.sum(track_candidates_chunk['track_quality'] & (quality_mask << 24) == (quality_mask << 24)))

            good_track_candidates = track_candidates_chunk[good_track_selection]
            good_track_candidates = good_track_candidates[np.logical_and(good_track_candidates['column_dut_%d' % fit_dut] != 0., good_track_candidates['row_dut_%d' % fit_dut] != 0.)]  # take only tracks where actual dut has a hit, otherwise residual wrong

            # Prepare track hits array to be fitted
            n_fit_duts = bin(dut_selection).count("1")
            index, n_tracks = 0, good_track_candidates['event_number'].shape[0]  # Index of tmp track hits array
            track_hits = np.zeros((n_tracks, n_fit_duts, 3))
            dut_hits_t = np.zeros((n_tracks, 3))
            for dut_index in range(0, n_duts):  # Fill index loop of new array
                if (1 << dut_index) & dut_selection == (1 << dut_index):  # True if DUT is used in fit
                    xr = good_track_candidates['column_dut_%s' % dut_index] * rotations[fit_dut, 0, 0] + good_track_candidates['row_dut_%s' % dut_index] * rotations[fit_dut, 0, 1] + translations[fit_dut, 0]
                    yr = good_track_candidates['column_dut_%s' % dut_index] * rotations[fit_dut, 1, 0] + good_track_candidates['row_dut_%s' % dut_index] * rotations[fit_dut, 1, 1] + translations[fit_dut, 1]
                    xyz = np.column_stack((xr, yr, np.repeat(z_positions[dut_index], n_tracks)))
                    track_hits[:, index, :] = xyz
                    index += 1
                elif dut_index == fit_dut:
                    xr = good_track_candidates['column_dut_%s' % dut_index] * rotations[fit_dut, 0, 0] + good_track_candidates['row_dut_%s' % dut_index] * rotations[fit_dut, 0, 1] + translations[fit_dut, 0]
                    yr = good_track_candidates['column_dut_%s' % dut_index] * rotations[fit_dut, 1, 0] + good_track_candidates['row_dut_%s' % dut_index] * rotations[fit_dut, 1, 1] + translations[fit_dut, 1]
                    xyz = np.column_stack((xr, yr, np.repeat(z_positions[dut_index], n_tracks)))
                    dut_hits_t = xyz

            # Split data and fit on all available cores
            n_slices = cpu_count()
            slice_length = np.ceil(1. * n_tracks / n_slices).astype(np.int32)
            slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
            pool = Pool(n_slices)
            results = pool.map(_fit_tracks_loop, slices)
            pool.close()
            pool.join()

            # Store results
            offsets_t = np.concatenate([i[0] for i in results])  # merge offsets from all cores in results
            slopes_t = np.concatenate([i[1] for i in results])  # merge slopes from all cores in results
            if n_chunk == 0:
                offsets = offsets_t
                slopes = slopes_t
                dut_hits = dut_hits_t
            else:
                offsets = np.concatenate([offsets, offsets_t])
                slopes = np.concatenate([slopes, slopes_t])
                dut_hits = np.concatenate([dut_hits, dut_hits_t])
            n_chunk += 1

    return dut_hits, offsets, slopes


def align_by_residuals(track_candidates_file, geometry, z_positions, fit_duts, include_duts, output_pdf):
    logging.info('=== Fine align the DUT translation by residuals ===')

    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    nplanes = len(z_positions)
    translations = np.zeros((nplanes, 3))
    rotations = np.zeros((nplanes, 3, 3))

    for index in range(3):
        translations[:, index] = geometry['translation_%d' % index]
    for i in range(3):
        for j in range(3):
            rotations[:, i, j] = geometry['rotation_%d_%d' % (i, j)]

    corr_translation = np.zeros((nplanes, 2))

    output_fig = PdfPages(output_pdf) if output_pdf else None
    for actual_dut in fit_duts:
        hits, offset, slope = fit_tracks_align(track_candidates_file, z_positions, actual_dut, geometry, include_duts=include_duts, track_quality=1)
        intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane
        '''Local residuals '''
#         tmpc = hits[:, 0] * rotations[actual_dut, 0, 0] + hits[:, 1] * rotations[actual_dut, 0, 1] + translations[actual_dut, 0]
#         tmpr = hits[:, 0] * rotations[actual_dut, 1, 0] + hits[:, 1] * rotations[actual_dut, 1, 1] + translations[actual_dut, 1]
#         hits[:, 0] = tmpc
#         hits[:, 1] = tmpr
        '''Global residuals '''
        tmppc = intersection[:, 0] - translations[actual_dut, 0]
        tmppr = intersection[:, 1] - translations[actual_dut, 1]
        tmpc = tmppc * rotations[actual_dut, 0, 0] + tmppr * rotations[actual_dut, 1, 0]
        tmpr = tmppc * rotations[actual_dut, 0, 1] + tmppr * rotations[actual_dut, 1, 1]
        intersection[:, 0] = tmpc
        intersection[:, 1] = tmpr

        difference = hits - intersection
        logging.info('Calculate residuals for DUT %d', actual_dut)
        for i in range(2):  # col / row
            mean, rms = np.mean(difference[:, i]), np.std(difference[:, i])
            hist, edges = np.histogram(difference[:, i], range=(mean - 5. * rms, mean + 5. * rms), bins=1000)
            #hist, edges = np.histogram(difference[:, i], range=(-150, 150), bins=100)
            fit_ok = False
            coeff, var_matrix = None, None
            try:
                #coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
                coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), 0., 15.])
                corr_translation[actual_dut, i] = -coeff[1]

                fit_ok = True
            except:
                fit_ok = False

            if output_pdf is not False:
                plot_utils.plot_residuals(i, actual_dut, edges, hist, fit_ok, coeff, gauss, difference, var_matrix, output_fig=output_fig)

    if output_fig:
        output_fig.close()

    return corr_translation


def fine_alignment(input_track_candidates_file, alignment_file, z_positions, output_pdf, fit_duts=range(6), include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], ignore_duts=None, create_new_geometry=True):
    '''Takes the track candidates, and fits a track for each DUT using the neigbouring DUTs in an iterative way.
    Plots the residuals in x / y as a function of x / y to deduce rotation and translation parameters.
    These parameters are set in the aligment file and used to correct the hit positions in the track candidates array.

    Parameters
    ----------
    input_track_candidates_file : pytbales file
        The input file with the track candidates.
    alignment_file : pytables file
        The output file for geometry parameters.
    z_positions: iterable
        the positions along beam line of the planes
    output_pdf : pdf file
        File name for the alignment plots
    fit_duts: iterable
        which duts to align
    include_duts: iterable
        the relative dut positions of dut to use in the track fit. The position is relative to the actual dut the tracks are fitted for
    create_new_geometry: boolean
        if true a new geometry file is created with default geometry (all angles and translations to 0). If alignment_file exists it is overwritten
        if false the current one is used (then alignment_file must exist)
    '''
    logging.info('=== Fine align the DUTs using line fit residuals ===')

    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    def calculate_residuals_correlation(data1, data2, edges1, edges2, fit_dut, label, dofit=True):
        step = edges1[1] - edges1[0]
        nbins = len(edges1)
        resx = [[]] * nbins
        for ind in range(nbins):
            resx[ind] = []
        mean_fitted = np.zeros(nbins)
        sigma_fitted = np.zeros(nbins)
        mean_error_fitted = np.zeros(nbins)
        n_hits = np.zeros(nbins)

        for i, x in enumerate(data1):
            n = np.int((x - edges1[0]) / step)
            resx[n].append(data2[i])

        for n in range(nbins):
            if len(resx[n]) == 0:
                mean_fitted[n] = -1
                mean_error_fitted[n] = 1000
                n_hits[n] = 0
                continue
            p0 = [np.amax(resx[n]), 0, 10]
            hist, edges = np.histogram(resx[n], range=(edges2[0], edges2[-1]), bins=len(edges2))
            ed = (edges[:-1] + edges[1:]) / 2.
            try:
                coeff, var_matrix = curve_fit(gauss, ed, hist, p0=p0)
                mean_fitted[n] = coeff[1]
                mean_error_fitted[n] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
                n_hits[n] = len(resx[n])
            except RuntimeError:
                pass

        mean_fitted[~np.isfinite(mean_fitted)] = -1
        mean_error_fitted[mean_error_fitted > 1000] = 1000
        selected_data = np.where(np.logical_and(mean_fitted != -1, 1 > 0))[0]

        f = lambda x, c0: c0 + x
        if dofit:
            fit, pcov = curve_fit(f, edges1[selected_data], mean_fitted[selected_data])
        else:
            fit, pcov = None, None

        refit = True
        selected_data = np.ones_like(mean_fitted, dtype=np.bool)
#        x = np.arange(1.5, mean_fitted.shape[0] + 1.5)
        x = edges1
        while(refit):
            # print mean_fitted
            selected_data, fit, refit = plot_utils.plot_alignments_fine(x, mean_fitted, mean_error_fitted, n_hits, label, 'DUT%d' % fit_dut, data1, data2, edges1, edges2)
            x = x[selected_data]
            mean_fitted = mean_fitted[selected_data]
            mean_error_fitted = mean_error_fitted[selected_data]
            sigma_fitted = sigma_fitted[selected_data]
            n_hits = n_hits[selected_data]

        # return mean_fitted, selected_data, fit, pcov
        return fit

    def calculate_residuals_correlations_all(hits, offset, slope, actual_dut, z_positions, geometry, output_fig, use_duts=None, doFit=True):
        nplanes = len(z_positions)

        translations = np.zeros((nplanes, 3))
        rotations = np.zeros((nplanes, 3, 3))

        for index in range(3):
            translations[:, index] = geometry['translation_%d' % index]
        for i in range(3):
            for j in range(3):
                rotations[:, i, j] = geometry['rotation_%d_%d' % (i, j)]

        logging.info('Calculate residuals for DUT %d', actual_dut)

        fits = np.zeros((2, 2, 2))

        intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane
        tmpc = hits[:, 0] * rotations[actual_dut, 0, 0] + hits[:, 1] * rotations[actual_dut, 0, 1] + translations[actual_dut, 0]
        tmpr = hits[:, 0] * rotations[actual_dut, 1, 0] + hits[:, 1] * rotations[actual_dut, 1, 1] + translations[actual_dut, 1]
        hits[:, 0] = tmpc
        hits[:, 1] = tmpr

        difference = hits - intersection

        for i in range(2):  # col / row
            for j in range(2):
                mean, rms = np.mean(difference[:, j]), np.std(difference[:, j])
                _, xedges, yedges = np.histogram2d(hits[:, i], difference[:, j], bins=[100, 1000], range=[[np.amin(hits[:, i]), np.amax(hits[:, i])], [mean - 3 * rms, mean + 3 * rms]])
                #_, xedges, yedges = np.histogram2d(hits[:, i], difference[:, j], bins=[100, 150], range=[[np.amin(hits[:, i]), np.amax(hits[:, i])], [-150,150]])
                plot_utils.plot_residuals_correlations(i, j, actual_dut, xedges, yedges, hits[:, i], difference[:, j], output_fig)
                label = "Residual "
                if i == 0:
                    label += "col"
                else:
                    label += "row"
                label += " vs "
                if j == 0:
                    label += "col"
                else:
                    label += "row"

                if doFit == True:
                    fit = calculate_residuals_correlation(hits[:, i], difference[:, j], xedges, yedges, actual_dut, label, dofit=True)
                else:
                    fit = None
                fits[i, j] = fit
                #plot_utils.plot_residuals_correlations_fit(i, j, actual_dut, xedges, yedges, mean_fitted, selected_data, fit, pcov)

        if doFit == True:
            return fits
        else:
            return None

    def calculate_geopars_from_fit(fitpars, verbose=False):
        translations = np.zeros(2)
        angles = np.zeros(3)

        translations[0] = -fitpars[1, 0, 0]
        translations[1] = -fitpars[0, 1, 0]

        tangamma = fitpars[0, 1, 1] / (1 - np.abs(fitpars[0, 0, 1]))
        singamma = np.sign(tangamma) * sqrt(tangamma ** 2 / (1 + tangamma ** 2))
        angles[2] = asin(singamma)
        cosbeta = (1 - np.abs(fitpars[0, 0, 1])) / sqrt(1 - singamma ** 2)
        if cosbeta > 1:
            cosbeta = 1 - (cosbeta - 1)  # sure it is fine?
        angles[1] = asin(sqrt(1 - cosbeta ** 2))
        cosalpha = (-np.abs(fitpars[1, 1, 1]) - tangamma * fitpars[1, 0, 1] + 1) / (sqrt(1 - singamma ** 2) + singamma * tangamma)
        if cosalpha > 1:
            cosalpha = 1 - (cosalpha - 1)
        angles[0] = asin(sqrt(1 - cosalpha ** 2))
#        angles[0] = -asin(sinalpha)
        if verbose:
            print "Gamma: ", angles[2]
            print "Beta: ", angles[1]
            print "Alpha: ", angles[0]

        return translations, angles

    if create_new_geometry == True:
        geometry_utils.create_initial_geometry(alignment_file, z_positions)

    nplanes = len(z_positions)

    with tb.open_file(alignment_file, mode='r') as alignment:
        all_fits = np.zeros((nplanes, 2, 2, 2))
        corr_translations = np.zeros((nplanes, 2))
        corr_angles = np.zeros((nplanes, 3))

        geometry = alignment.root.Geometry[:]
        output_pdf0 = output_pdf[:-4]
        output_pdf0 += "_stage0.pdf"
        ''' Fit residual vs position plots and correct angle '''
        with PdfPages(output_pdf0) as output_fig0:
            for fit_dut in fit_duts:
                dut_hits, offsets, slopes = fit_tracks_align(input_track_candidates_file, z_positions, fit_dut, geometry, include_duts=include_duts, ignore_duts=ignore_duts, track_quality=1)
                fit_plot_tracks_slopes(slopes, fit_dut, output_fig0)
                fit = calculate_residuals_correlations_all(dut_hits, offsets, slopes, fit_dut, z_positions, geometry, output_fig0)
                if fit is not None:
                    all_fits[fit_dut] = fit
                    corr_translations[fit_dut], corr_angles[fit_dut] = calculate_geopars_from_fit(fit)
                    logging.info('Dut %d: translation correction: %.1f, %.1f', fit_dut, corr_translations[fit_dut, 0], corr_translations[fit_dut, 1])
                    logging.info('Dut %d: angles correction: %.5f, %.5f, %.5f', fit_dut, corr_angles[fit_dut, 0], corr_angles[fit_dut, 1], corr_angles[fit_dut, 2])

    ''' Update geometry file'''
    for dut in range(nplanes):
        geometry_utils.update_translation_val(alignment_file, dut, corr_translations[dut, 0], corr_translations[dut, 1], mode="Relative")
        geometry_utils.update_rotation_angles(alignment_file, dut, corr_angles[dut], mode="Relative")

    ''' Correct eventual translation misalignment: shift positions by the residuals'''
    output_pdf0_res = output_pdf0[:-4] + "_residuals.pdf"
    corr_trans = align_by_residuals(input_track_candidates_file, geometry, z_positions, fit_duts, include_duts, output_pdf0_res)
    for dut in range(nplanes):
        geometry_utils.update_translation_val(alignment_file, dut, corr_trans[dut, 0], corr_trans[dut, 1], mode="Relative")

    ''' Remake the plots to verify the alignment'''
    with tb.open_file(alignment_file, mode='r') as alignment:
        geometry = alignment.root.Geometry[:]
        output_pdf1 = output_pdf[:-4]
        output_pdf1 += "_stage1.pdf"
        output_pdf1_res = output_pdf1[:-4] + "_residuals.pdf"

        with PdfPages(output_pdf1) as output_fig1:
            for fit_dut in fit_duts:
                dut_hits, offsets, slopes = fit_tracks_align(input_track_candidates_file, z_positions, fit_dut, geometry, include_duts=include_duts, ignore_duts=ignore_duts, track_quality=1)
                fit_plot_tracks_slopes(slopes, fit_dut, output_fig1)
                calculate_residuals_correlations_all(dut_hits, offsets, slopes, fit_dut, z_positions, geometry, output_fig1, doFit=False)

    align_by_residuals(input_track_candidates_file, geometry, z_positions, fit_duts, include_duts, output_pdf1_res)


def merge_cluster_data(input_cluster_files, input_alignment_file, output_tracklets_file, pixel_size, chunk_size=4999999):
    '''Takes the cluster from all cluster files and merges them into one big table onto the event number.
    Empty entries are signaled with charge = 0. The position is referenced from the correlation data to the first plane.
    Function uses easily several GB of RAM. If memory errors occur buy a better PC or chunk this function.

    Parameters
    ----------
    input_cluster_files : list of pytables files
        File name of the input cluster files with correlation data.
    input_alignment_file : pytables file
        File name of the input aligment data.
    output_tracklets_file : pytables file
        File name of the output tracklet file.
    limit_events : int
        Limit events to givien number. Only events with hits are counted. If None or 0, all events will be taken.
    chunk_size: int
        Defines the amount of in RAM data. The higher the more RAM is used and the faster this function works.
    '''
    logging.info('=== Merge cluster from %d DUTSs to tracklets ===', len(input_cluster_files))

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        alignment = in_file_h5.root.Alignment[:]

    # Create result array description, depends on the number of DUTs
    description = [('event_number', np.int64)]
    for index, _ in enumerate(input_cluster_files):
        description.append(('column_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('row_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('charge_dut_%d' % index, np.float))
    description.extend([('track_quality', np.uint32), ('n_tracks', np.uint8)])

    start_indices = [0] * len(input_cluster_files)  # Store the loop indices for speed up
    start_indices_2 = [0] * len(input_cluster_files)  # Additional indices for second loop

    # Merge the cluster data from different DUTs into one table
    with tb.open_file(output_tracklets_file, mode='w') as out_file_h5:
        tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracklets', description=np.zeros((1,), dtype=description).dtype, title='Tracklets', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
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
                tracklets_array = np.zeros((common_event_numbers.shape[0],), dtype=description)  # Result array to be filled. For no hit: column = row = 0
                # Fill result array with DUT 0 data
                tracklets_array['event_number'] = common_event_numbers[:]
                actual_cluster = analysis_utils.map_cluster(common_event_numbers, cluster_dut_0)
                selection = actual_cluster['mean_column'] != 0  # Add only real hits, 0 is a virtual hit
                tracklets_array['column_dut_0'][selection] = pixel_size[0][0] * actual_cluster['mean_column'][selection]  # Convert channel indices to um
                tracklets_array['row_dut_0'][selection] = pixel_size[0][1] * actual_cluster['mean_row'][selection]  # Convert channel indices to um
                tracklets_array['charge_dut_0'][selection] = actual_cluster['charge'][selection]

                # Fill result array with other DUT data
                # Second loop: get the cluster from all files and merge them to the common event number
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_cluster, start_indices_2[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start=start_indices_2[dut_index], start_event_number=actual_start_event_number, stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size):  # Loop over the cluster in the actual cluster file in chunks
                            actual_cluster = analysis_utils.map_cluster(common_event_numbers, actual_cluster)
                            selection = actual_cluster['mean_column'] != 0  # Add only real hits, 0 is a virtual hit
                            actual_mean_column = pixel_size[dut_index][0] * actual_cluster['mean_column'][selection]  # Convert channel indices to um
                            actual_mean_row = pixel_size[dut_index][1] * actual_cluster['mean_row'][selection]  # Convert channel indices to um
                            # Apply alignment information
                            c0 = alignment[alignment['dut_x'] == (dut_index)]['c0']
                            c1 = alignment[alignment['dut_x'] == (dut_index)]['c1']
                            tracklets_array['column_dut_%d' % (dut_index)][selection] = (c1[0] * actual_mean_column + c0[0])
                            tracklets_array['row_dut_%d' % (dut_index)][selection] = (c1[1] * actual_mean_row + c0[1])
                            tracklets_array['charge_dut_%d' % (dut_index)][selection] = actual_cluster['charge'][selection]

                np.nan_to_num(tracklets_array)
                tracklets_table.append(tracklets_array)
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(index)
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


def optimize_hit_alignment(input_tracklets_file, input_alignment_file, fraction=10):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).

    Parameters
    ----------
    input_tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    aligment_file : string
        Input file name with alignment data
    use_fraction : float
        Use only every fraction-th hit for the alignment correction. For speed up. 1 means all hits are used
    '''
    logging.info('=== Optimize hit alignment ===')
    with tb.open_file(input_tracklets_file, mode="r+") as in_file_h5:
        particles = in_file_h5.root.Tracklets[:]
        with tb.open_file(input_alignment_file, 'r+') as alignment_file_h5:
            alignment_data = alignment_file_h5.root.Alignment[:]
            n_duts = int(alignment_data.shape[0] / 2)
            for table_column in in_file_h5.root.Tracklets.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    actual_dut = int(re.findall(r'\d+', table_column)[-1])
                    ref_dut_column = re.sub(r'\d+', '0', table_column)
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
