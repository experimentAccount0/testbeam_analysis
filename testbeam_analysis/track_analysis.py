''' Track finding and fitting functions are listed here.'''
from __future__ import division

import logging
import progressbar
from multiprocessing import Pool, cpu_count
from math import sqrt
from cmath import log
import re
import os.path

from pykalman.standard import KalmanFilter
import tables as tb
import numpy as np
from scipy.optimize import curve_fit
from numba import njit
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis import plot_utils
from testbeam_analysis import analysis_utils
from testbeam_analysis import geometry_utils


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def find_tracks(input_tracklets_file, input_alignment_file, output_track_candidates_file, event_range=None, chunk_size=1000000):
    '''Takes first DUT track hit and tries to find matching hits in subsequent DUTs.
    The output is the same array with resorted hits into tracks. A track quality is set to
    be able to cut on good (less scattered) tracks.
    This function is uses numba to increase the speed on the inner loop (_find_tracks_loop()).

    This function can also be called on TrackCandidates arrays. That is usefull if an additional alignment step
    was done and the track finding has to be repeated.

    Parameters
    ----------
    input_tracklets_file : string
        Input file name with merged cluster hit table from all DUTs (tracklets file)
        Or track candidates file.
    input_alignment_file : string
        File containing the alignment information
    output_track_candidates_file : string
        Output file name for track candidate array
    '''
    logging.info('=== Find tracks ===')

    # Get alignment errors from file
    with tb.open_file(input_alignment_file, mode='r') as in_file_h5:
        try:
            correlations = in_file_h5.root.Alignment[:]
            n_duts = correlations.shape[0]
            logging.info('Taking correlation cut values from alignment')
            column_sigma = correlations['correlation_x']
            row_sigma = correlations['correlation_y']
        except tb.exceptions.NoSuchNodeError:
            logging.info('Taking correlation cut values from prealignment')
            correlations = in_file_h5.root.Prealignment[:]
            n_duts = int(correlations.shape[0] / 2 + 1)
            column_sigma = np.zeros(shape=n_duts)
            row_sigma = np.zeros(shape=n_duts)
            column_sigma[0], row_sigma[0] = 0., 0.  # DUT0 has no correlation error
            for index in range(1, n_duts):
                column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
                row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    with tb.open_file(input_tracklets_file, mode='r') as in_file_h5:
        try:  # First try:  normal tracklets assumed
            tracklets_node = in_file_h5.root.Tracklets
        except tb.exceptions.NoSuchNodeError:
            try:  # Second try: normal track candidates assumed
                tracklets_node = in_file_h5.root.TrackCandidates
                output_track_candidates_file = os.path.splitext(output_track_candidates_file)[0] + '_2.h5'
                logging.info('Additional find track run on track candidates file %s', input_tracklets_file)
                logging.info('Output file with new track candidates file %s', output_track_candidates_file)
            except tb.exceptions.NoSuchNodeError:  # Last try: prealigned tracklets from coarse alignment assumed
                tracklets_node = in_file_h5.root.Tracklets_prealignment
        with tb.open_file(output_track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=tracklets_node.dtype, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            for tracklets_data_chunk, index in analysis_utils.data_aligned_at_events(tracklets_node, chunk_size=chunk_size):
                tracklets_data_chunk = tracklets_data_chunk.view(np.recarray)

                # Prepare data for track finding, create arrays for x, y position and charge data
                tr_x = tracklets_data_chunk['x_dut_0']
                tr_y = tracklets_data_chunk['y_dut_0']
                tr_z = tracklets_data_chunk['z_dut_0']
                tr_charge = tracklets_data_chunk['charge_dut_0']
                for dut_index in range(n_duts - 1):
                    tr_x = np.vstack((tr_x, tracklets_data_chunk['x_dut_%d' % (dut_index + 1)]))
                    tr_y = np.vstack((tr_y, tracklets_data_chunk['y_dut_%d' % (dut_index + 1)]))
                    tr_z = np.vstack((tr_z, tracklets_data_chunk['z_dut_%d' % (dut_index + 1)]))
                    tr_charge = np.vstack((tr_charge, tracklets_data_chunk['charge_dut_%d' % (dut_index + 1)]))
                tr_x = np.transpose(tr_x)
                tr_y = np.transpose(tr_y)
                tr_z = np.transpose(tr_z)
                tr_charge = np.transpose(tr_charge)

                tracklets_data_chunk.track_quality = np.zeros(shape=tracklets_data_chunk.shape[0])  # If find tracks is called on already found tracks the track quality has to be reset

                # Perform the track finding with jitted loop
                tracklets_data_chunk, tr_x, tr_y, tr_z, tr_charge = _find_tracks_loop(tracklets_data_chunk, tr_x, tr_y, tr_z, tr_charge, column_sigma, row_sigma)

                # Merge result data from arrays into one recarray
                combined = np.column_stack((tracklets_data_chunk.event_number, tr_x, tr_y, tr_z, tr_charge, tracklets_data_chunk.track_quality, tracklets_data_chunk.n_tracks))
                combined = np.core.records.fromarrays(combined.transpose(), dtype=tracklets_data_chunk.dtype)

                track_candidates.append(combined)


def find_tracks_corr(input_tracklets_file, input_alignment_file, output_track_candidates_file, pixel_size):
    '''Takes first DUT track hit and tries to find matching hits in subsequent DUTs.
    Works on corrected Tracklets file, is identical to the find_tracks function.
    The output is the same array with resorted hits into tracks. A track quality is given to
    be able to cut on good tracks.
    This function is slow since the main loop happens in Python (< 1e5 tracks / second)
    but does the track finding loop on all cores in parallel (_find_tracks_loop()).
    Does the same as find_tracks function but works on a corrected TrackCandidates file (optimize_track_aligment)

    Parameters
    ----------
    input_tracklets_file : string
        Input file name with merged cluster hit table from all DUTs
    input_alignment_file : string
        File containing the alignment information
    output_track_candidates_file : string
        Output file name for track candidate array
    '''
    logging.info('=== Build tracks from TrackCandidates ===')

    # Get alignment errors from file
    with tb.open_file(input_alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] // 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    with tb.open_file(input_tracklets_file, mode='r') as in_file_h5:
        tracklets = in_file_h5.root.TrackCandidates
        n_duts = sum(['column' in col for col in tracklets.dtype.names])

        # prepare data for track finding, create arrays for column, row and charge data
        tracklets = tracklets[:].view(np.recarray)
        tr_column = tracklets['column_dut_0']
        tr_row = tracklets['row_dut_0']
        tr_charge = tracklets['charge_dut_0']
        for dut_index in range(n_duts - 1):
            tr_column = np.vstack((tr_column, tracklets['column_dut_%d' % (dut_index + 1)]))
            tr_row = np.vstack((tr_row, tracklets['row_dut_%d' % (dut_index + 1)]))
            tr_charge = np.vstack((tr_charge, tracklets['charge_dut_%d' % (dut_index + 1)]))
        tr_column = np.transpose(tr_column)
        tr_row = np.transpose(tr_row)
        tr_charge = np.transpose(tr_charge)

        # Perform the track finding with jitted loop
        tracklets, tr_column, tr_row, tr_charge = _find_tracks_loop(tracklets, tr_column, tr_row, tr_charge, column_sigma, row_sigma)

        # Merge result data from arrays into one recarray
        combined = np.column_stack((tracklets.event_number, tr_column, tr_row, tr_charge, tracklets.track_quality, tracklets.n_tracks))
        combined = np.core.records.fromarrays(combined.transpose(), dtype=tracklets.dtype)

        with tb.open_file(output_track_candidates_file, mode='w') as out_file_h5:
            track_candidates2 = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=in_file_h5.root.TrackCandidates.description, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            track_candidates2.append(combined)


def optimize_track_alignment(input_track_candidates_file, input_alignment_file, fraction=1, correlated_only=False):
    '''This step should not be needed but alignment checks showed an offset between the hit positions after alignment
    especially for DUTs that have a flipped orientation. This function corrects for the offset (c0 in the alignment).
    Does the same as optimize_hit_alignment but works on TrackCandidates file.
    If optimize_track_aligment is used track quality can change and must be calculated again from corrected data (use find_tracks_corr).


    Parameters
    ----------
    input_track_candidates_file : string
        Input file name with merged cluster hit table from all DUTs
    input_alignment_file : string
        Input file with alignment data
    use_fraction : float
        Use only every fraction-th hit for the alignment correction. For speed up. 1 means all hits are used
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('=== Optimize track alignment ===')
    with tb.open_file(input_track_candidates_file, mode="r+") as in_file_h5:
        particles = in_file_h5.root.TrackCandidates[:]
        with tb.open_file(input_alignment_file, 'r+') as alignment_file_h5:
            alignment_data = alignment_file_h5.root.Alignment[:]
            n_duts = alignment_data.shape[0] / 2
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    actual_dut = int(re.findall(r'\d+', table_column)[-1])
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Optimize alignment for % s', table_column)
                    particle_selection = particles[::fraction][np.logical_and(particles[::fraction][ref_dut_column] > 0, particles[::fraction][table_column] > 0)]  # only select events with hits in both DUTs
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


def check_track_alignment(trackcandidates_files, output_pdf, combine_n_hits=10000000, correlated_only=False, track_quality=None):
    '''Takes the tracklet array and plots the difference of column/row position of each DUT against the reference DUT0
    for every combine_n_events. If the alignment worked the median has to be around 0 and should not change with time
    (with the event number).
    Does the same as check_hit_alignment but works on TrackCandidates file.

    Parameters
    ----------
    input_track_candidates_file : string
        Input file name with merged cluster hit table from all DUTs
    output_pdf : pdf file name object
    combine_n_hits : int
        The number of events to combine for the hit position check
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 5-sigma of the correlation
        2: The track hits in DUT and reference are within 2-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    '''
    logging.info('=== Check TrackCandidates Alignment ===')
    with tb.open_file(trackcandidates_files, mode="r") as in_file_h5:
        with PdfPages(output_pdf) as output_fig:
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'dut' in table_column and 'dut_0' not in table_column and 'charge' not in table_column:
                    dut_index = int(table_column[-1])  # DUT index of actual DUT data
                    median, mean, std, alignment, correlation = [], [], [], [], []
                    ref_dut_column = table_column[:-1] + '0'
                    logging.info('Check alignment for % s', table_column)
                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.TrackCandidates.shape[0], term_width=80)
                    progress_bar.start()
                    for index in range(0, in_file_h5.root.TrackCandidates.shape[0], combine_n_hits):
                        particles = in_file_h5.root.TrackCandidates[index:index + combine_n_hits:5]  # take every 10th hit
                        particles = particles[np.logical_and(particles[ref_dut_column] > 0, particles[table_column] > 0)]  # only select events with hits in both DUTs
                        if correlated_only is True:
                            particles = particles[particles['track_quality'] & (1 << (24 + dut_index)) == (1 << (24 + dut_index))]
                        if track_quality:
                            particles = particles[particles['track_quality'] & (1 << (track_quality * 8 + dut_index)) == (1 << (track_quality * 8 + dut_index))]
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


def fit_tracks(input_track_candidates_file, input_alignment_file, output_tracks_file, fit_duts=None, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], track_quality=1, max_tracks=None, output_pdf_file=None, use_correlated=False, chunk_size=1000000):
    '''Fits a line through selected DUT hits for selected DUTs. The selection criterion for the track candidates to fit is the track quality and the maximum number of hits per event.
    The fit is done for specified DUTs only (fit_duts). This DUT is then not included in the fit (include_duts). Bad DUTs can be always ignored in the fit (ignore_duts).

    Parameters
    ----------
    input_track_candidates_file : string
        file name with the track candidates table
    input_alignment_file : pytables file
        File name of the input aligment data
    output_tracks_file : string
        file name of the created track file having the track table
    fit_duts : iterable
        the duts to fit tracks for. If None all duts are used
    ignore_duts : iterable
        the duts that are not taken in a fit. Needed to exclude bad planes from track fit. Also included Duts are ignored!
    include_duts : iterable
        the relative dut positions of dut to use in the track fit. The position is relative to the actual dut the tracks are fitted for
        e.g. actual track fit dut = 2, include_duts = [-3, -2, -1, 1] means that duts 0, 1, 3 are used for the track fit
    max_tracks : int, None
        only events with tracks <= max tracks are taken
    track_quality : int
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 2-sigma of the correlation
        2: The track hits in DUT and reference are within 1-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    pixel_size : iterable, (x dimensions, y dimension)
        the size in um of the pixels, needed for chi2 calculation
    output_pdf_file : pdf file name object
        Name of plots pdf file
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''

    logging.info('=== Fit tracks ===')

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        alignment = in_file_h5.root.Alignment[:]

    def create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts):
        # Define description
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('x_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('y_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('z_dut_%d' % index, np.float))
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
            tracks_array['x_dut_%d' % index] = good_track_candidates['x_dut_%d' % index]
            tracks_array['y_dut_%d' % index] = good_track_candidates['y_dut_%d' % index]
            tracks_array['z_dut_%d' % index] = good_track_candidates['z_dut_%d' % index]
            tracks_array['charge_dut_%d' % index] = good_track_candidates['charge_dut_%d' % index]
        for dimension in range(3):
            tracks_array['offset_%d' % dimension] = offsets[:, dimension]
            tracks_array['slope_%d' % dimension] = slopes[:, dimension]
        tracks_array['track_chi2'] = chi2s

        return tracks_array

    with PdfPages(output_pdf_file) as output_fig:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            with tb.open_file(output_tracks_file, mode='w') as out_file_h5:
                n_duts = sum(['charge' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                fit_duts = fit_duts if fit_duts else range(n_duts)
                for fit_dut in fit_duts:  # Loop over the DUTs where tracks shall be fitted for
                    logging.info('Fit tracks for DUT %d', fit_dut)

                    dut_position = np.array([alignment[fit_dut]['translation_x'], alignment[fit_dut]['translation_y'], alignment[fit_dut]['translation_z']])
                    rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[fit_dut]['alpha'],
                                                                     beta=alignment[fit_dut]['beta'],
                                                                     gamma=alignment[fit_dut]['gamma'])
                    basis_global = rotation_matrix.T.dot(np.eye(3))  # TODO: why transposed?
                    dut_plane_normal = basis_global[2]

                    tracklets_table = None
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

                        # Prepare track hits array to be fitted
                        n_fit_duts = bin(dut_selection).count("1")
                        index, n_tracks = 0, good_track_candidates['event_number'].shape[0]  # Index of tmp track hits array
                        track_hits = np.zeros((n_tracks, n_fit_duts, 3))
                        for dut_index in range(0, n_duts):  # Fill index loop of new array
                            if (1 << dut_index) & dut_selection == (1 << dut_index):  # True if DUT is used in fit
                                xyz = np.column_stack((good_track_candidates['x_dut_%s' % dut_index], good_track_candidates['y_dut_%s' % dut_index], good_track_candidates['z_dut_%s' % dut_index]))
                                track_hits[:, index, :] = xyz
                                index += 1

                        # Split data and fit on all available cores
                        n_slices = cpu_count()
                        slice_length = np.ceil(1. * n_tracks / n_slices).astype(np.int32)
                        slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
                        pool = Pool(n_slices)
                        results = pool.map(_fit_tracks_loop, slices)
                        pool.close()
                        pool.join()
                        del track_hits

                        # Store results
                        offsets = np.concatenate([i[0] for i in results])  # Merge offsets from all cores in results
                        slopes = np.concatenate([i[1] for i in results])  # Merge slopes from all cores in results
                        chi2s = np.concatenate([i[2] for i in results])  # Merge chi2 from all cores in results

                        # Set the offset to the track intersection with the tilded plane
                        offsets = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                                                line_directions=slopes,
                                                                                                position_plane=dut_position,
                                                                                                normal_plane=dut_plane_normal)

                        tracks_array = create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts)

                        if tracklets_table is None:
                            tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                        tracklets_table.append(tracks_array)

                        # Plot chi2 distribution
                        plot_utils.plot_track_chi2(chi2s, fit_dut, output_fig)


def fit_tracks_kalman(input_track_candidates_file, output_tracks_file, geometry_file, z_positions, fit_duts=None, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], track_quality=1, max_tracks=None, output_pdf=None, use_correlated=False, method="Interpolation", pixel_size=[], chunk_size=1000000):
    '''Fits a line through selected DUT hits for selected DUTs. The selection criterion for the track candidates to fit is the track quality and the maximum number of hits per event.
    The fit is done for specified DUTs only (fit_duts). This DUT is then not included in the fit (include_duts). Bad DUTs can be always ignored in the fit (ignore_duts).

    Parameters
    ----------
    input_track_candidates_file : string
        file name with the track candidates table
    output_tracks_file : string
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
    output_pdf : pdf file name object
        if None plots are printed to screen
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    method: string
        Defines the method for hit prediction:
            "Interpolation": chi2 minimization with straight line
            "Kalman": Kalman filter
    geometry_file: the file containing the geometry parameters (relative translation and angles)
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
        for index in range(n_duts):
            description.append(('predicted_x%d' % index, np.float))
        for index in range(n_duts):
            description.append(('predicted_y%d' % index, np.float))
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
            intersection = offsets + slopes / slopes[:, 2, np.newaxis] * (z_positions[index] - offsets[:, 2, np.newaxis])  # intersection track with DUT plane
            tracks_array['predicted_x%d' % index] = intersection[:, 0]
            tracks_array['predicted_y%d' % index] = intersection[:, 1]
        for dimension in range(3):
            tracks_array['offset_%d' % dimension] = offsets[:, dimension]
            tracks_array['slope_%d' % dimension] = slopes[:, dimension]
        tracks_array['track_chi2'] = chi2s

        return tracks_array

    def create_results_array_kalman(good_track_candidates, track_estimates, chi2s, n_duts):
        # Define description
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('column_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('row_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('charge_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('predicted_x%d' % index, np.float))
        for index in range(n_duts):
            description.append(('predicted_y%d' % index, np.float))
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
            tracks_array['predicted_x%d' % index] = track_estimates[:, index, 0]
            tracks_array['predicted_y%d' % index] = track_estimates[:, index, 1]
        tracks_array['track_chi2'] = chi2s

        return tracks_array

    method = method.lower()
    if method != "interpolation" and method != "kalman":
        raise ValueError('Method "%s" not recognized!' % method)
    if method == "kalman" and not pixel_size:
        raise ValueError('Kalman filter requires to provide pixel size for error measurement matrix covariance!')

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            with tb.open_file(output_tracks_file, mode='w') as out_file_h5:
                n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                fit_duts = fit_duts if fit_duts else range(n_duts)
                for fit_dut in fit_duts:  # Loop over the DUTs where tracks shall be fitted for
                    logging.info('Fit tracks for DUT %d', fit_dut)
                    tracklets_table = None
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

                        good_track_candidates_chunk = track_candidates_chunk[good_track_selection]

                        # Prepare track hits array to be fitted
                        n_fit_duts = bin(dut_selection).count("1")
                        index, n_tracks = 0, good_track_candidates_chunk['event_number'].shape[0]  # Index of tmp track hits array

                        translations, rotations = geometry_utils.recontruct_geometry_from_file(geometry_file)

                        if method == "interpolation":
                            track_hits = np.zeros((n_tracks, n_fit_duts, 3))
                        elif method == "kalman":
                            track_hits = np.zeros((n_tracks, n_duts, 3))
                        for dut_index in range(0, n_duts):  # Fill index loop of new array
                            if method == "interpolation" and (1 << dut_index) & dut_selection == (1 << dut_index):  # True if DUT is used in fit
                                xr = good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 0, 0] + good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 0, 1] + translations[dut_index, 0]
                                yr = good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 1, 1] + good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 1, 0] + translations[dut_index, 1]
                                xyz = np.column_stack((xr, yr, np.repeat(z_positions[dut_index], n_tracks)))
                                track_hits[:, index, :] = xyz
                                index += 1
                            elif method == "kalman":
                                if (1 << dut_index) & dut_selection == (1 << dut_index):  # TOCHECK! Not used = masked, OK, but also DUT must be masked...
                                    # xyz = np.column_stack(np.ma.array((good_track_candidates_chunk['column_dut_%s' % dut_index], good_track_candidates_chunk['row_dut_%s' % dut_index], np.repeat(z_positions[dut_index], n_tracks))))
                                    xr = good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 0, 0] + good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 0, 1] + translations[dut_index, 0]
                                    yr = good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 1, 1] + good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 1, 0] + translations[dut_index, 1]
                                    xyz = np.column_stack(np.ma.array((xr, yr, np.repeat(z_positions[dut_index], n_tracks))))
                                else:
                                    xr = good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 0, 0] + good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 0, 1] + translations[dut_index, 0]
                                    yr = good_track_candidates_chunk['row_dut_%s' % dut_index] * rotations[dut_index, 1, 1] + good_track_candidates_chunk['column_dut_%s' % dut_index] * rotations[dut_index, 1, 0] + translations[dut_index, 1]
                                    xyz = np.column_stack(np.ma.array((xr, yr, np.repeat(z_positions[dut_index], n_tracks)), mask=np.ones((n_tracks, 3))))
                                track_hits[:, index, :] = xyz
                                index += 1

                        # Split data and fit on all available cores
                        n_slices = cpu_count()
                        slice_length = np.ceil(1. * n_tracks / n_slices).astype(np.int32)

                        pool = Pool(n_slices)
                        if method == "interpolation":
                            slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
                            results = pool.map(_fit_tracks_loop, slices)
                        elif method == "kalman":
                            slices = [track_hits[i:i + slice_length] for i in range(0, n_tracks, slice_length)]
                            # arg = (slices, pixel_size, z_positions)
                            args = [(track_hits[i:i + slice_length], pixel_size, z_positions) for i in range(0, n_tracks, slice_length)]
                            # args = [(data_files[i], n_pixels[i][0], n_pixels[i][1], 16, 14) for i in range(0, len(data_files))]
                            results = pool.map(_function_wrapper_fit_tracks_kalman_loop, args)
                        pool.close()
                        pool.join()

                        # Store results
                        if method == "interpolation":
                            offsets = np.concatenate([i[0] for i in results])  # merge offsets from all cores in results
                            slopes = np.concatenate([i[1] for i in results])  # merge slopes from all cores in results
                            chi2s = np.concatenate([i[2] for i in results])  # merge chi2 from all cores in results
                            tracks_array = create_results_array(good_track_candidates_chunk, slopes, offsets, chi2s, n_duts)
                            if not tracklets_table:
                                tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                            tracklets_table.append(tracks_array)
                            for i in range(2):
                                mean, rms = np.mean(slopes[:, i]), np.std(slopes[:, i])
                                hist, edges = np.histogram(slopes[:, i], range=(mean - 5. * rms, mean + 5. * rms), bins=1000)
                                fit_ok = False
                                try:
                                    coeff, var_matrix = curve_fit(gauss, edges[:-1], hist, p0=[np.amax(hist), mean, rms])
                                    fit_ok = True
                                except:
                                    fit_ok = False
                                plot_utils.plot_tracks_parameter(slopes, edges, i, hist, fit_ok, coeff, gauss, var_matrix, output_fig, fit_dut, parName='Slope')
                                meano, rmso = np.mean(offsets[:, i]), np.std(offsets[:, i])
                                histo, edgeso = np.histogram(offsets[:, i], range=(meano - 5. * rmso, meano + 5. * rmso), bins=1000)
                                fit_ok = False
                                try:
                                    coeffo, var_matrixo = curve_fit(gauss, edgeso[:-1], histo, p0=[np.amax(histo), meano, rmso])
                                    fit_ok = True
                                except:
                                    fit_ok = False
                                plot_utils.plot_tracks_parameter(offsets, edgeso, i, histo, fit_ok, coeffo, gauss, var_matrixo, output_fig, fit_dut, parName='Offset')
                        elif method == "kalman":
                            track_estimates = np.concatenate([i[0] for i in results])  # merge predicted x,y pos from all cores in results
                            chi2s = np.concatenate([i[1] for i in results])  # merge chi2 from all cores in results
                            tracks_array = create_results_array_kalman(good_track_candidates_chunk, track_estimates, chi2s, n_duts)
                            if not tracklets_table:
                                tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_Kalman_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks Kalman-smoothed for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                            tracklets_table.append(tracks_array)

                        # Plot chi2 distribution
                        plot_utils.plot_track_chi2(chi2s, fit_dut, output_fig)


# Helper functions that are not meant to be called during analysis
@njit
def _set_dut_track_quality(tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma):
    # Set track quality of actual DUT from actual DUT hit
    column, row = tr_column[track_index][dut_index], tr_row[track_index][dut_index]
    if row != 0:  # row = 0 is no hit
        actual_track.track_quality |= (1 << dut_index)  # Set track with hit
        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
        if column_distance < 1 * actual_column_sigma and row_distance < 1 * actual_row_sigma:  # High quality track hits
            actual_track.track_quality |= (65793 << dut_index)
        elif column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # Low quality track hits
            actual_track.track_quality |= (257 << dut_index)
    else:
        actual_track.track_quality &= (~(65793 << dut_index))  # Unset track quality


@njit
def _reset_dut_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, hit_index, actual_column_sigma, actual_row_sigma):
    # Recalculate track quality of already assigned hit, needed if hits are swapped
    first_dut_index = _get_first_dut_index(tr_column, hit_index)

    actual_track_column, actual_track_row = tr_column[hit_index][first_dut_index], tr_row[hit_index][first_dut_index]
    actual_track = tracklets[hit_index]
    column, row = tr_column[hit_index][dut_index], tr_row[hit_index][dut_index]

    actual_track.track_quality &= ~(65793 << dut_index)  # Reset track quality to zero

    if row != 0:  # row = 0 is no hit
        actual_track.track_quality |= (1 << dut_index)  # Set track with hit
        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
        if column_distance < 1 * actual_column_sigma and row_distance < 1 * actual_row_sigma:  # High quality track hits
            actual_track.track_quality |= (65793 << dut_index)
        elif column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # Low quality track hits
            actual_track.track_quality |= (257 << dut_index)


@njit
def _get_first_dut_index(tr_column, index):
    ''' Returns the first DUT that has a hit for the track at index '''
    dut_index = 0
    for dut_index in range(tr_column.shape[1]):  # Loop over duts, to get first DUT hit of track
        if tr_column[index][dut_index] != 0:
            break
    return dut_index


@njit
def _swap_hits(tr_column, tr_row, tr_z, tr_charge, track_index, dut_index, hit_index, column, row, z, charge):
    #     print 'Swap hits', tr_column[track_index][dut_index], tr_column[hit_index][dut_index]
    tmp_column, tmp_row, tmp_z, tmp_charge = tr_column[track_index][dut_index], tr_row[track_index][dut_index], tr_z[track_index][dut_index], tr_charge[track_index][dut_index]
    tr_column[track_index][dut_index], tr_row[track_index][dut_index], tr_z[track_index][dut_index], tr_charge[track_index][dut_index] = column, row, z, charge
    tr_column[hit_index][dut_index], tr_row[hit_index][dut_index], tr_z[hit_index][dut_index], tr_charge[hit_index][dut_index] = tmp_column, tmp_row, tmp_z, tmp_charge


@njit
def _find_tracks_loop(tracklets, tr_column, tr_row, tr_z, tr_charge, column_sigma, row_sigma):
    ''' Complex loop to resort the tracklets array inplace to form track candidates. Each track candidate
    is given a quality identifier. Not ment to be called stand alone.
    Optimizations included to make it compile with numba. Can be called from
    several real threads if they work on different areas of the array'''
    n_duts = tr_column.shape[1]
    actual_event_number = tracklets[0].event_number

    # Numba uses c scopes, thus define all used variables here
    n_actual_tracks = 0
    track_index, actual_hit_track_index = 0, 0  # Track index of table and first track index of actual event
    column, row = 0., 0.
    actual_track_column, actual_track_row = 0., 0.
    column_distance, row_distance = 0., 0.
    hit_distance = 0.

    for track_index, actual_track in enumerate(tracklets):  # Loop over all possible tracks
#         print '== ACTUAL TRACK  ==', track_index
        # Set variables for new event
        if actual_track.event_number != actual_event_number:  # Detect new event
            actual_event_number = actual_track.event_number
            for i in range(n_actual_tracks):  # Set number of tracks of previous event
                tracklets[track_index - 1 - i].n_tracks = n_actual_tracks
            n_actual_tracks = 0
            actual_hit_track_index = track_index

        n_actual_tracks += 1
        reference_hit_set = False  # The first real hit (column, row != 0) is the reference hit of the actual track
        n_track_hits = 0

        for dut_index in range(n_duts):  # loop over all DUTs in the actual track
            actual_column_sigma, actual_row_sigma = column_sigma[dut_index], row_sigma[dut_index]

#             print '== ACTUAL DUT  ==', dut_index

            if not reference_hit_set and tr_row[track_index][dut_index] != 0:  # Search for first DUT that registered a hit (row != 0)
                actual_track_column, actual_track_row = tr_column[track_index][dut_index], tr_row[track_index][dut_index]
                reference_hit_set = True
                tracklets[track_index].track_quality |= (65793 << dut_index)  # First track hit has best quality by definition
                n_track_hits += 1
#                 print 'ACTUAL REFERENCE HIT', actual_track_column, actual_track_row
            elif reference_hit_set:  # First hit found, now find best (closest) DUT hit
                shortest_hit_distance = -1  # The shortest hit distance to the actual hit; -1 means not assigned
                for hit_index in range(actual_hit_track_index, tracklets.shape[0]):  # Loop over all not sorted hits of actual DUT
                    if tracklets[hit_index].event_number != actual_event_number:  # Abort condition
                        break
                    column, row, z, charge = tr_column[hit_index][dut_index], tr_row[hit_index][dut_index], tr_z[hit_index][dut_index], tr_charge[hit_index][dut_index]
                    if row != 0:  # Check for hit (row != 0)
                        # Calculate the hit distance of the actual DUT hit towards the actual reference hit
                        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
                        hit_distance = sqrt(column_distance * column_distance + row_distance * row_distance)
                        # Calculate the hit distance of the actual assigned DUT hit towards the actual reference hit
                        current_column_distance, current_row_distance = abs(tr_column[track_index][dut_index] - actual_track_column), abs(tr_row[track_index][dut_index] - actual_track_row)
                        current_hit_distance = sqrt(current_column_distance * current_column_distance + current_row_distance * current_row_distance)  # The hit distance of the actual assigned hit
                        if shortest_hit_distance < 0 or hit_distance < shortest_hit_distance:  # Check if the hit is closer to reference hit
#                             print 'FOUND MATCHING HIT', column, row
                            if track_index != hit_index:  # Check if hit swapping is needed
                                if track_index > hit_index:  # Check if hit is already assigned to other track
#                                     print 'BUT HIT ALREADY ASSIGNED TO TRACK', hit_index
                                    first_dut_index = _get_first_dut_index(tr_column, hit_index)  # Get reference DUT index of other track
                                    # Calculate hit distance to reference hit of other track
                                    column_distance_old, row_distance_old = abs(column - tr_column[hit_index][first_dut_index]), abs(row - tr_row[hit_index][first_dut_index])
                                    hit_distance_old = sqrt(column_distance_old * column_distance_old + row_distance_old * row_distance_old)
                                    if current_hit_distance < hit_distance:
#                                         print 'CURRENT ASSIGNED HIT FITS BETTER, DO NOT SWAP', hit_index
                                        continue
                                    if hit_distance > hit_distance_old:  # Only take hit if it fits better to actual track; otherwise leave it with other track
#                                         print 'IT FIT BETTER WITH OLD TRACK, DO NOT SWAP', hit_index
                                        continue
#                                 print 'SWAP HIT'
                                _swap_hits(tr_column, tr_row, tr_z, tr_charge, track_index, dut_index, hit_index, column, row, z, charge)
                                if track_index > hit_index:  # Check if hit is already assigned to other track
                                    #                                     print 'RESET DUT TRACK QUALITY'
                                    _reset_dut_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, hit_index, actual_column_sigma, actual_row_sigma)
                            shortest_hit_distance = hit_distance
                            n_track_hits += 1

#             if reference_dut_index == n_duts - 1:  # Special case: If there is only one hit in the last DUT, check if this hit fits better to any other track of this event
#                 pass
#             print 'SET DUT TRACK QUALITY'
            _set_dut_track_quality(tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)

#         print 'TRACK', track_index
#         for dut_index in range(n_duts):
#             print tr_row[track_index][dut_index],
#         print
        # Set number of tracks of last event
        for i in range(n_actual_tracks):
            tracklets[track_index - i].n_tracks = n_actual_tracks

    return tracklets, tr_column, tr_row, tr_z, tr_charge


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


def _function_wrapper_find_tracks_loop(args):  # Needed for multiprocessing call with arguments
    return _find_tracks_loop(*args)


def _function_wrapper_fit_tracks_kalman_loop(args):  # Needed for multiprocessing call with arguments
    return _fit_tracks_kalman_loop(*args)


def _kalman_fit_3d(hits, transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance, observation_offset, initial_state_mean, initial_state_covariance, sigma):

    kf = KalmanFilter(
        transition_matrix, observation_matrix, transition_covariance, observation_covariance, transition_offset, observation_offset,
        initial_state_mean, initial_state_covariance
    )

    nplanes = hits.shape[0]
    meas = np.c_[hits, np.zeros((nplanes))]
    # kf = kf.em(meas, n_iter=5)
    smoothed_state_estimates = kf.smooth(meas)[0]

    chi2 = 0
    chi2 += np.sum(np.dot(np.square(meas[:, 0] - smoothed_state_estimates[:, 0]), np.square(1 / sigma)), dtype=np.double)

    return smoothed_state_estimates[:, 0], chi2


def _fit_tracks_kalman_loop(track_hits, pitches, plane_pos):
    nplanes = track_hits.shape[1]
    # TODO: from parameter
    thickness_si = [50. for _ in range(nplanes)]
    thickness_al = [600. for _ in range(nplanes)]
    thicknesses = thickness_si + thickness_al
    sigma = np.dot([pitches[i] for i in range(nplanes)], 1 / sqrt(12.))  # Resolution of each telescope plane
    # TOFIX
    tmp_plane_pos = [plane_pos[i] for i in range(nplanes)]
    # TOFIX : these two happens in case more planes are provided than dut files...

    ''' Calculations for multiple scattering'''
    X0si = 93600.  # radiation length in Silicon = 9.36 cm (Google Introduction to silicon trackers at LHC - TDX)
    X0al = 89000.  # in Aluminum
    energy = 100000.  # energy in MeV
    mass = 0.511  # mass in MeV
    momentum = sqrt(energy * energy - mass * mass)
    # beta = momentum / energy
    x0s = np.dot(thickness_si, 1 / X0si) + np.dot(thickness_al, 1 / X0al)
    thetas = np.zeros(nplanes, dtype=np.double)
    for i, xx in enumerate(x0s):
        thetat = ((13.6 / momentum) * sqrt(xx) * (1 + 0.038 * log(xx)))  # from formula
        thetas[i] = thetat.real
    print("Thetas: ")
    print(thetas)

    '''Kalman filter parameters'''
    transition_matrix = np.zeros((nplanes, 2, 2))
    for i, z in enumerate(plane_pos):
        transition_matrix[i] = [[1, 0], [0, 1]]
        if i < nplanes - 1:
            transition_matrix[i, 0, 1] = plane_pos[i + 1] - z
        else:
            transition_matrix[i, 0, 1] = transition_matrix[i - 1, 0, 1]
        if i >= nplanes - 1:
            break  # TOFIX

    transition_covariance = np.zeros((nplanes, 2, 2))
    for j, t in enumerate(thetas):
        transition_covariance[j] = [[t * t * thicknesses[j] * thicknesses[j] / 3, t * t * thicknesses[j] / 2], [t * t * thicknesses[j] / 2, t * t]]  # from some calculations

    transition_offset = [0, 0]
    # transition_covariance = [[theta * theta * thickness * thickness / 3, theta * theta * thickness / 2], [theta * theta * thickness / 2, theta * theta]]  # from some calculations
    observation_matrix = [[1, 0], [0, 0]]
    observation_offset = transition_offset
    observation_covariance_x = np.zeros((sigma.shape[0], 2, 2))
    observation_covariance_y = np.zeros((sigma.shape[0], 2, 2))
    observation_covariance_x[:, 0, 0] = np.square(sigma[:, 0])
    observation_covariance_y[:, 0, 0] = np.square(sigma[:, 1])

    ''' Initial state: first hit with slope 0, error: its sigma and a large one for slope '''
    initial_state_covariance_x = [[sigma[0, 0] ** 2, 0], [0, 0.01]]
    initial_state_covariance_y = [[sigma[0, 1] ** 2, 0], [0, 0.01]]

    track_estimates = np.zeros((track_hits.shape))

    chi2 = np.zeros((track_hits.shape[0]))

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        initial_state_mean_x = [actual_hits[0, 0], 0]
        initial_state_mean_y = [actual_hits[0, 1], 0]
        track_estimates_x, chi2x = _kalman_fit_3d(actual_hits[:, 0], transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance_x, observation_offset, initial_state_mean_x, initial_state_covariance_x, sigma[:, 0])
        track_estimates_y, chi2y = _kalman_fit_3d(actual_hits[:, 1], transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance_y, observation_offset, initial_state_mean_y, initial_state_covariance_y, sigma[:, 1])
        chi2[index] = chi2x + chi2y
        track_estimates[index, :, 0] = track_estimates_x
        track_estimates[index, :, 1] = track_estimates_y
        track_estimates[index, :, 2] = tmp_plane_pos

    return track_estimates, chi2
