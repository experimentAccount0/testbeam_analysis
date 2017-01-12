''' Track finding and fitting functions are listed here.'''
from __future__ import division

import logging
from multiprocessing import Pool, cpu_count
from math import sqrt
import progressbar
import os
from collections import Iterable

import tables as tb
import numpy as np
from numba import njit
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import geometry_utils


def find_tracks(input_tracklets_file, input_alignment_file, output_track_candidates_file, min_cluster_distance=False, chunk_size=1000000):
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
    min_cluster_distance : iterable, boolean
        A minimum distance all track cluster have to be apart, otherwise the complete event is flagged to have merged tracks (n_tracks = -1).
        This is needed to get a correct efficiency number, since assigning the same cluster to several tracks is not implemented and error prone.
        If it is true the std setting of 200 um is used. Otherwise a distance in um for each DUT has to be given.
        e.g.: For two devices: min_cluster_distance = (50, 250)
        If false the cluster distance is not considered.
        The events where any plane does have hits < min_cluster_distance is flagged with n_tracks = -1
    '''
    logging.info('=== Find tracks ===')

    # Get alignment errors from file
    with tb.open_file(input_alignment_file, mode='r') as in_file_h5:
        try:
            raise tb.exceptions.NoSuchNodeError  # FIXME: sigma is to small after alignment, track finding with tracks instead of correlation needed
            correlations = in_file_h5.root.Alignment[:]
            n_duts = correlations.shape[0]
            logging.info('Taking correlation cut values from alignment')
            column_sigma = correlations['correlation_x']
            row_sigma = correlations['correlation_y']
        except tb.exceptions.NoSuchNodeError:
            logging.info('Taking correlation cut values from pre-alignment')
            correlations = in_file_h5.root.PreAlignment[:]
            n_duts = correlations.shape[0]
            if min_cluster_distance is True:
                min_cluster_distance = np.array([(200.)] * n_duts)
            elif min_cluster_distance is False:
                min_cluster_distance = np.zeros(n_duts)
            else:
                min_cluster_distance = np.array(min_cluster_distance)
            column_sigma = np.zeros(shape=n_duts)
            row_sigma = np.zeros(shape=n_duts)
            column_sigma[0], row_sigma[0] = 0., 0.  # DUT0 has no correlation error
            for index in range(1, n_duts):
                column_sigma[index] = correlations[index]['column_sigma']
                row_sigma[index] = correlations[index]['row_sigma']

    with tb.open_file(input_tracklets_file, mode='r') as in_file_h5:
        try:  # First try:  normal tracklets assumed
            tracklets_node = in_file_h5.root.Tracklets
        except tb.exceptions.NoSuchNodeError:
            try:  # Second try: normal track candidates assumed
                tracklets_node = in_file_h5.root.TrackCandidates
                logging.info('Additional find track run on track candidates file %s', input_tracklets_file)
                logging.info('Output file with new track candidates file %s', output_track_candidates_file)
            except tb.exceptions.NoSuchNodeError:  # Last try: not used yet
                raise
        with tb.open_file(output_track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=tracklets_node.dtype, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=tracklets_node.shape[0], term_width=80)
            progress_bar.start()

            for tracklets_data_chunk, index in analysis_utils.data_aligned_at_events(tracklets_node, chunk_size=chunk_size):
                # Prepare hit data for track finding, create temporary arrays for x, y, z position and charge data
                # This is needed to call a numba jitted function, since the number of DUTs is not fixed and thus the data format
                tr_x = tracklets_data_chunk['x_dut_0']
                tr_y = tracklets_data_chunk['y_dut_0']
                tr_z = tracklets_data_chunk['z_dut_0']
                tr_charge = tracklets_data_chunk['charge_dut_0']
                for dut_index in range(1, n_duts):
                    tr_x = np.column_stack((tr_x, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
                    tr_y = np.column_stack((tr_y, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
                    tr_z = np.column_stack((tr_z, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
                    tr_charge = np.column_stack((tr_charge, tracklets_data_chunk['charge_dut_%d' % (dut_index)]))

                tracklets_data_chunk['track_quality'] = np.zeros(shape=tracklets_data_chunk.shape[0])  # If find tracks is called on already found tracks the track quality has to be reset

                # Perform the track finding with jitted loop
                _find_tracks_loop(tracklets_data_chunk, tr_x, tr_y, tr_z, tr_charge, column_sigma, row_sigma, min_cluster_distance)

                # Merge result data from arrays into one recarray
                combined = np.column_stack((tracklets_data_chunk['event_number'], tr_x, tr_y, tr_z, tr_charge, tracklets_data_chunk['track_quality'], tracklets_data_chunk['n_tracks']))
                combined = np.core.records.fromarrays(combined.transpose(), dtype=tracklets_data_chunk.dtype)

                track_candidates.append(combined)
                progress_bar.update(index)
            progress_bar.finish()


def fit_tracks(input_track_candidates_file, input_alignment_file, output_tracks_file, fit_duts=None, selection_hit_duts=None, selection_fit_duts=None, exclude_dut_hit=True, selection_track_quality=1, max_tracks=None, force_prealignment=False, use_correlated=False, min_track_distance=False, chunk_size=1000000):
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
    selection_hit_duts : iterable, or iterable of iterable
        The duts that are required to have a hit with the given track quality. Otherwise the track is omitted
        If None: require all DUTs to have a hit, but if exclude_dut_hit = True do not use actual fit_dut.
        If iterable: use selection for all devices, e.g.: Require hit in DUT 0, and 3: selection_hit_duts = (0, 3)
        If iterable of iterable: define dut with hits for all devices seperately.
        E.g: for 3 devices: selection_hit_duts = ((1, 2), (0, 1, 2), (0, 1))
    selection_fit_duts : iterable, or iterable of iterable or None
        If None: selection_hit_duts are used for fitting
        Cannot define DUTs that are not in selection_hit_duts.
        E.g.: Require hits in DUT0, DUT1, DUT3, DUT4 but do not use DUT3 in the fit:
        selection_hit_duts = (0, 1, 3, 4)
        selection_fit_duts = (0, 1, 4)
    exclude_dut_hit: boolean
        Set to not require a hit in the actual fit DUT (e.g.: for uncontrained residuals)
        False: Just use all devices as specified in selection_hit_duts.
        True: Do not take the DUT hit for track selection / fitting, even if specified in selection_hit_duts
    max_tracks : int, None
        only events with tracks <= max tracks are taken
    selection_track_quality : int, iterable
        One number valid for all DUTs or an iterable with a number for each DUT.
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 2-sigma of the correlation
        2: The track hits in DUT and reference are within 1-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    pixel_size : iterable, (x dimensions, y dimension)
        the size in um of the pixels, needed for chi2 calculation
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    min_track_distance : iterable, boolean
        A minimum distance all track intersection at the DUT have to be apart, otherwise these tracks are deleted.
        This is needed to get a correct efficiency number, since assigning the same cluster to several tracks is error prone and will not be implemented.
        If it is true the std setting of 200 um is used. Otherwise a distance in um for each DUT has to be given.
        e.g.: For two devices: min_track_distance = (50, 250)
        If false the track distance is not considered.
    '''

    logging.info('=== Fit tracks ===')

    # Load alignment data
    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        z_positions = in_file_h5.root.PreAlignment[:]['z']
        if not use_prealignment:
            try:
                alignment = in_file_h5.root.Alignment[:]
                use_prealignment = False
            except tb.exceptions.NodeError:
                z_positions = in_file_h5.root.PreAlignment[:]['z']
                use_prealignment = True
        n_duts = z_positions.shape[0]

    if use_prealignment:
        logging.info('Use pre-alignment data')
    else:
        logging.info('Use alignment data')

    # Create track, hit selection
    if not selection_hit_duts:  # If None: use all DUTs
        selection_hit_duts = [i for i in range(n_duts)]

    if not isinstance(selection_track_quality, Iterable):
        selection_track_quality = [selection_track_quality for _ in selection_hit_duts]

    # Std. case: use all DUTs that are required to have a hit for track fitting
    if not selection_fit_duts:
        selection_fit_duts = selection_hit_duts

    if not isinstance(selection_fit_duts, Iterable):
        selection_fit_duts = [selection_hit_duts for _ in range(n_duts)]

    # Convert potential selection valid for all duts to required selection for each DUT
    try:
        iter(selection_hit_duts[0])
    except TypeError:  # not iterable
        selection_hit_duts = [selection_hit_duts for _ in range(n_duts)]
    try:
        iter(selection_fit_duts[0])
    except:
        selection_fit_duts = [selection_fit_duts for _ in range(n_duts)]
    try:
        iter(selection_track_quality[0])
    except:
        selection_track_quality = [selection_track_quality for _ in range(n_duts)]

    if len(selection_hit_duts) != len(selection_track_quality):
        raise ValueError('The length of the hit dut selection has to be equal to the quality selection!')

    for (sel_1, sel_2) in zip(selection_hit_duts, selection_fit_duts):
        if not all(x in sel_1 for x in sel_2):
            raise NotImplementedError('All DUTs defined in selection_fit_duts have to be defined in selection_hit_duts!')

    # Special mode: use all DUTs in the fit and the selections are all the same --> the data does only have to be fitted once
    if not exclude_dut_hit and all(x == selection_hit_duts[0] for x in selection_hit_duts) and all(x == selection_fit_duts[0] for x in selection_fit_duts) and all(x == selection_track_quality[0] for x in selection_track_quality):
        same_tracks_for_all_duts = True
    else:
        same_tracks_for_all_duts = False

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
        description.extend([('track_chi2', np.uint32), ('track_quality', np.uint32), ('n_tracks', np.int8)])

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

    def store_track_data(fit_dut, min_track_distance):  # Set the offset to the track intersection with the tilted plane and store the data
        if not use_prealignment:  # Deduce plane orientation in 3D for track extrapolation; not needed if rotation info is not available (e.g. only prealigned data)
            dut_position = np.array([alignment[fit_dut]['translation_x'], alignment[fit_dut]['translation_y'], alignment[fit_dut]['translation_z']])
            rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[fit_dut]['alpha'],
                                                             beta=alignment[fit_dut]['beta'],
                                                             gamma=alignment[fit_dut]['gamma'])
            basis_global = rotation_matrix.T.dot(np.eye(3))  # TODO: why transposed?
            dut_plane_normal = basis_global[2]
        else:  # Pre-alignment does not set any plane rotations thus plane normal = (0, 0, 1) and position = (0, 0, z)
            dut_position = np.array([0., 0., z_positions[fit_dut]])
            dut_plane_normal = np.array([0., 0., 1.])

        # Set the offset to the track intersection with the tilted plane
        actual_offsets = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                          line_directions=slopes,
                                                                          position_plane=dut_position,
                                                                          normal_plane=dut_plane_normal)

        tracks_array = create_results_array(good_track_candidates, slopes, actual_offsets, chi2s, n_duts)

        try:  # Check if table exists already, than append data
            tracklets_table = out_file_h5.get_node('/Tracks_DUT_%d' % fit_dut)
        except tb.NoSuchNodeError:  # Table does not exist, thus create new
            tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

        # Remove tracks that are too close when extrapolated to the actual DUT
        # All merged track are signaled by n_tracks = -1
        actual_min_track_distance = min_track_distance[fit_dut]
        if actual_min_track_distance > 0:
            _find_merged_tracks(tracks_array, actual_min_track_distance)
            selection = tracks_array['n_tracks'] > 0
            logging.info('Removed %d merged tracks (%1.1f%%)', np.count_nonzero(~selection), float(np.count_nonzero(~selection)) / selection.shape[0] * 100.)
            tracks_array = tracks_array[selection]

        tracklets_table.append(tracks_array)

        # Plot chi2 distribution
        plot_utils.plot_track_chi2(chi2s, fit_dut, output_fig)

    def select_data(dut_index):  # Select track by and DUT hits to use

        dut_selection = 0  # DUTs to be used in the fit
        dut_fit_selection = 0  # DUT to use in fit
        info_str_hit = ''  # For info output
        info_str_fit = ''  # For info output

        for hit_dut in selection_hit_duts[dut_index]:
            if exclude_dut_hit and hit_dut == dut_index:
                continue
            dut_selection |= ((1 << hit_dut))
            info_str_hit += 'DUT%d ' % (hit_dut)
        for selected_fit_dut in selection_fit_duts[dut_index]:
            if exclude_dut_hit and selected_fit_dut == dut_index:
                continue
            dut_fit_selection |= ((1 << selected_fit_dut))
            info_str_fit += 'DUT%d ' % (selected_fit_dut)

        if same_tracks_for_all_duts:
            logging.info('Fit tracks for all DUTs at the same time!')

        logging.info('Use %d DUTs for track selection: %s', bin(dut_selection)[2:].count("1"), info_str_hit)
        logging.info("Use %d DUTs for track fit: %s", bin(dut_fit_selection)[2:].count("1"), info_str_fit)

        track_quality_mask = 0
        for index, dut in enumerate(selection_hit_duts[dut_index]):
            if exclude_dut_hit and dut == dut_index:
                continue
            for quality in range(3):
                if quality <= selection_track_quality[dut_index][index]:
                    track_quality_mask |= ((1 << dut) << quality * 8)
        logging.info("Use track quality: %s", str(selection_track_quality[dut_index])[1:-1])
        return dut_selection, dut_fit_selection, track_quality_mask, same_tracks_for_all_duts

    pool = Pool()
    with PdfPages(output_tracks_file[:-3] + '.pdf') as output_fig:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            try:  # If file exists already delete it first
                os.remove(output_tracks_file)
            except OSError:
                pass
            with tb.open_file(output_tracks_file, mode='a') as out_file_h5:  # Append mode to be able to append to existing tables; file is created here since old file is deleted
                n_duts = sum(['charge' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                fit_duts = fit_duts if fit_duts is not None else range(n_duts)  # Std. setting: fit tracks for all DUTs

                if min_track_distance is True:
                    min_track_distance = np.array([(200.)] * n_duts)
                elif min_track_distance is False:
                    min_track_distance = np.zeros(n_duts)
                elif isinstance(min_track_distance, (int, float)):
                    min_track_distance = np.array([(min_track_distance)] * n_duts)
                else:
                    min_track_distance = np.array(min_track_distance)

                for fit_dut in fit_duts:  # Loop over the DUTs where tracks shall be fitted for
                    logging.info('Fit tracks for DUT %d', fit_dut)

                    dut_selection, dut_fit_selection, track_quality_mask, same_tracks_for_all_duts = select_data(fit_dut)
                    n_fit_duts = bin(dut_fit_selection)[2:].count("1")
                    if n_fit_duts < 2:
                        logging.warning('Insufficient track hits to do the fit (< 2). Omit DUT %d', fit_dut)
                        continue

                    progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.TrackCandidates.shape[0], term_width=80)
                    progress_bar.start()

                    for track_candidates_chunk, index_candidates in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates, chunk_size=chunk_size):

                        # Select tracks based on the dut that are required to have a hit (dut_selection) with a certain quality (track_quality)

                        good_track_selection = (track_candidates_chunk['track_quality'] & track_quality_mask) == track_quality_mask

                        n_track_cut = good_track_selection.shape[0] - np.count_nonzero(good_track_selection)

                        good_track_selection = np.logical_and(good_track_selection, track_candidates_chunk['n_tracks'] > 0)  # n_tracks < 0 means merged cluster, omit these to allow valid efficiency calculation

                        n_merged_cut = good_track_selection.shape[0] - np.count_nonzero(good_track_selection) - n_track_cut

                        if max_tracks:  # Option to neglect events with too many hits
                            good_track_selection = np.logical_and(good_track_selection, track_candidates_chunk['n_tracks'] <= max_tracks)
                            n_tracks_cut = good_track_selection.shape[0] - np.count_nonzero(good_track_selection) - n_track_cut - n_merged_cut
                            logging.info('Removed %d tracks candidates (%d tracks due to quality, %d tracks due to merged cluster, %d # tracks), %.1f%% ',
                                         good_track_selection.shape[0] - np.count_nonzero(good_track_selection),
                                         n_track_cut,
                                         n_merged_cut,
                                         n_tracks_cut,
                                         (1. - float(np.count_nonzero(good_track_selection) / float(good_track_selection.shape[0]))) * 100.)
                        else:
                            logging.info('Removed %d tracks candidates (%d tracks due to quality, %d tracks due to merged cluster), %.1f%% ',
                                         good_track_selection.shape[0] - np.count_nonzero(good_track_selection),
                                         n_track_cut,
                                         n_merged_cut,
                                         (1. - float(np.count_nonzero(good_track_selection) / float(good_track_selection.shape[0]))) * 100.)

                        if use_correlated:  # Reduce track selection to correlated DUTs only
                            good_track_selection &= (track_candidates_chunk['track_quality'] & (dut_selection << 24) == (dut_selection << 24))
                            logging.info('Removed %d tracks candidates due to correlated cuts', good_track_selection.shape[0] - np.sum(track_candidates_chunk['track_quality'] & (dut_selection << 24) == (dut_selection << 24)))

                        good_track_candidates = track_candidates_chunk[good_track_selection]

                        # Prepare track hits array to be fitted
                        index, n_tracks = 0, good_track_candidates['event_number'].shape[0]  # Index of tmp track hits array
                        track_hits = np.empty((n_tracks, n_fit_duts, 3))
                        track_hits[:] = np.nan

                        for dut_index in range(0, n_duts):  # Fill index loop of new array
                            if ((1 << dut_index) & dut_fit_selection) == (1 << dut_index):  # True if DUT is used in fit
                                xyz = np.column_stack((good_track_candidates['x_dut_%s' % dut_index], good_track_candidates['y_dut_%s' % dut_index], good_track_candidates['z_dut_%s' % dut_index]))
                                track_hits[:, index, :] = xyz
                                index += 1

                        # Split data and fit on all available cores
                        n_slices = cpu_count()
                        slices = np.array_split(track_hits, n_slices)
                        results = pool.map(_fit_tracks_loop, slices)
                        del track_hits

                        # Store results
                        offsets = np.concatenate([i[0] for i in results])  # Merge offsets from all cores in results
                        slopes = np.concatenate([i[1] for i in results])  # Merge slopes from all cores in results
                        chi2s = np.concatenate([i[2] for i in results])  # Merge chi2 from all cores in results

                        # Store the data
                        if not same_tracks_for_all_duts:  # Check if all DUTs were fitted at once
                            store_track_data(fit_dut, min_track_distance)
                        else:
                            for fit_dut in fit_duts:
                                store_track_data(fit_dut, min_track_distance)

                        progress_bar.update(index_candidates)
                    progress_bar.finish()
                    if same_tracks_for_all_duts:  # Stop fit Dut loop since all DUTs were fitted at once
                        break
    pool.close()
    pool.join()


# Helper functions that are not meant to be called during analysis
@njit
def _set_dut_track_quality(tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma):
    # Set track quality of actual DUT from actual DUT hit
    column, row = tr_column[track_index][dut_index], tr_row[track_index][dut_index]
    if not np.isnan(row):  # row = nan is no hit
        actual_track['track_quality'] |= (1 << dut_index)  # Set track with hit
        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
        if column_distance < 1 * actual_column_sigma and row_distance < 1 * actual_row_sigma:  # High quality track hits
            actual_track['track_quality'] |= (65793 << dut_index)
        elif column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # Low quality track hits
            actual_track['track_quality'] |= (257 << dut_index)
    else:
        actual_track['track_quality'] &= (~(65793 << dut_index))  # Unset track quality


@njit
def _reset_dut_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, hit_index, actual_column_sigma, actual_row_sigma):
    # Recalculate track quality of already assigned hit, needed if hits are swapped
    first_dut_index = _get_first_dut_index(tr_column, hit_index)

    actual_track_column, actual_track_row = tr_column[hit_index][first_dut_index], tr_row[hit_index][first_dut_index]
    actual_track = tracklets[hit_index]
    column, row = tr_column[hit_index][dut_index], tr_row[hit_index][dut_index]

    actual_track['track_quality'] &= ~(65793 << dut_index)  # Reset track quality to zero

    if not np.isnan(row):  # row = nan is no hit
        actual_track['track_quality'] |= (1 << dut_index)  # Set track with hit
        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
        if column_distance < 1 * actual_column_sigma and row_distance < 1 * actual_row_sigma:  # High quality track hits
            actual_track['track_quality'] |= (65793 << dut_index)
        elif column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # Low quality track hits
            actual_track['track_quality'] |= (257 << dut_index)


@njit
def _get_first_dut_index(tr_column, index):
    ''' Returns the first DUT that has a hit for the track at index '''
    dut_index = 0
    for dut_index in range(tr_column.shape[1]):  # Loop over duts, to get first DUT hit of track
        if not np.isnan(tr_column[index][dut_index]):
            break
    return dut_index


@njit
def _swap_hits(tr_column, tr_row, tr_z, tr_charge, track_index, dut_index, hit_index, column, row, z, charge):
    #     print 'Swap hits', tr_column[track_index][dut_index], tr_column[hit_index][dut_index]
    tmp_column, tmp_row, tmp_z, tmp_charge = tr_column[track_index][dut_index], tr_row[track_index][dut_index], tr_z[track_index][dut_index], tr_charge[track_index][dut_index]
    tr_column[track_index][dut_index], tr_row[track_index][dut_index], tr_z[track_index][dut_index], tr_charge[track_index][dut_index] = column, row, z, charge
    tr_column[hit_index][dut_index], tr_row[hit_index][dut_index], tr_z[hit_index][dut_index], tr_charge[hit_index][dut_index] = tmp_column, tmp_row, tmp_z, tmp_charge


@njit
def _set_n_tracks(start_index, stop_index, tracklets, n_actual_tracks, tr_column, tr_row, min_cluster_distance, n_duts):
    if start_index < 0:
        start_index = 0

    if n_actual_tracks > 1:  # Only if the event has more than one track check the min_cluster_distance
        for dut_index in range(n_duts):
            if min_cluster_distance[dut_index] != 0:  # Check if minimum track distance evaluation is set, 0 is no mimimum track distance cut
                for i in range(start_index, stop_index):  # Loop over all event hits
                    actual_column, actual_row = tr_column[i][dut_index], tr_row[i][dut_index]
                    if np.isnan(actual_column):  # Omit virtual hit
                        continue
                    for j in range(i + 1, stop_index):  # Loop over other event hits
                        if sqrt((actual_column - tr_column[j][dut_index]) * (actual_column - tr_column[j][dut_index]) + (actual_row - tr_row[j][dut_index]) * (actual_row - tr_row[j][dut_index])) < min_cluster_distance[dut_index]:
                            for i in range(start_index, stop_index):  # Set number of tracks of this event to -1 to signal merged hits, thus merged tracks
                                tracklets[i]['n_tracks'] = -1
                            return

    # Called if no merged track is found
    for i in range(start_index, stop_index):  # Set number of tracks of previous event
        tracklets[i]['n_tracks'] = n_actual_tracks


@njit
def _find_tracks_loop(tracklets, tr_column, tr_row, tr_z, tr_charge, column_sigma, row_sigma, min_cluster_distance):
    ''' Complex loop to resort the tracklets array inplace to form track candidates. Each track candidate
    is given a quality identifier. Each hit is put to the best fitting track. Tracks are assumed to have
    no big angle, otherwise this approach does not work.
    Optimizations included to make it compile with numba. Can be called from
    several real threads if they work on different areas of the array'''
    n_duts = tr_column.shape[1]
    actual_event_number = tracklets[0]['event_number']

    # Numba uses c scopes, thus define all used variables here
    n_actual_tracks = 0
    track_index, actual_hit_track_index = 0, 0  # Track index of table and first track index of actual event
    column, row = np.nan, np.nan
    actual_track_column, actual_track_row = 0., 0.
    column_distance, row_distance = 0., 0.
    hit_distance = 0.

    for track_index, actual_track in enumerate(tracklets):  # Loop over all possible tracks
        #         print '== ACTUAL TRACK  ==', track_index
        # Set variables for new event
        if actual_track['event_number'] != actual_event_number:  # Detect new event
            actual_event_number = actual_track['event_number']
# for i in range(n_actual_tracks):  # Set number of tracks of previous event
#                 print 'old', track_index - 1 - i
# print 'track_index', track_index
#             tracklets[track_index - 1 - i]['n_tracks'] = n_actual_tracks
            _set_n_tracks(start_index=track_index - n_actual_tracks,
                          stop_index=track_index, tracklets=tracklets,
                          n_actual_tracks=n_actual_tracks,
                          tr_column=tr_column,
                          tr_row=tr_row,
                          min_cluster_distance=min_cluster_distance,
                          n_duts=n_duts)
            n_actual_tracks = 0
            actual_hit_track_index = track_index

        n_actual_tracks += 1
        reference_hit_set = False  # The first real hit (column, row != nan) is the reference hit of the actual track
        n_track_hits = 0

        for dut_index in range(n_duts):  # loop over all DUTs in the actual track
            actual_column_sigma, actual_row_sigma = column_sigma[dut_index], row_sigma[dut_index]

            # Calculate the hit distance of the actual assigned DUT hit towards the actual reference hit
            current_column_distance, current_row_distance = abs(tr_column[track_index][dut_index] - actual_track_column), abs(tr_row[track_index][dut_index] - actual_track_row)
            current_hit_distance = sqrt(current_column_distance * current_column_distance + current_row_distance * current_row_distance)  # The hit distance of the actual assigned hit
            if np.isnan(tr_column[track_index][dut_index]):  # No hit at the actual position
                current_hit_distance = -1  # Signal no hit

#             print '== ACTUAL DUT  ==', dut_index

            if not reference_hit_set and not np.isnan(tr_row[track_index][dut_index]):  # Search for first DUT that registered a hit
                actual_track_column, actual_track_row = tr_column[track_index][dut_index], tr_row[track_index][dut_index]
                reference_hit_set = True
                tracklets[track_index]['track_quality'] |= (65793 << dut_index)  # First track hit has best quality by definition
                n_track_hits += 1
#                 print 'ACTUAL REFERENCE HIT', actual_track_column, actual_track_row
            elif reference_hit_set:  # First hit found, now find best (closest) DUT hit
                shortest_hit_distance = -1  # The shortest hit distance to the actual hit; -1 means not assigned
                for hit_index in range(actual_hit_track_index, tracklets.shape[0]):  # Loop over all not sorted hits of actual DUT
                    if tracklets[hit_index]['event_number'] != actual_event_number:  # Abort condition
                        break
                    column, row, z, charge = tr_column[hit_index][dut_index], tr_row[hit_index][dut_index], tr_z[hit_index][dut_index], tr_charge[hit_index][dut_index]
                    if not np.isnan(row):  # Check for hit
                        # Calculate the hit distance of the actual DUT hit towards the actual reference hit
                        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
                        hit_distance = sqrt(column_distance * column_distance + row_distance * row_distance)
                        if shortest_hit_distance < 0 or hit_distance < shortest_hit_distance:  # Check if the hit is closer to reference hit
                            #                             print 'FOUND MATCHING HIT', column, row
                            if track_index != hit_index:  # Check if hit swapping is needed
                                if track_index > hit_index:  # Check if hit is already assigned to other track
                                    #                                     print 'BUT HIT ALREADY ASSIGNED TO TRACK', hit_index
                                    first_dut_index = _get_first_dut_index(tr_column, hit_index)  # Get reference DUT index of other track
                                    # Calculate hit distance to reference hit of other track
                                    column_distance_old, row_distance_old = abs(column - tr_column[hit_index][first_dut_index]), abs(row - tr_row[hit_index][first_dut_index])
                                    hit_distance_old = sqrt(column_distance_old * column_distance_old + row_distance_old * row_distance_old)
                                    if current_hit_distance > 0 and current_hit_distance < hit_distance:  # Check if actual assigned hit is better
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

# if reference_dut_index == n_duts - 1:  # Special case: If there is only one hit in the last DUT, check if this hit fits better to any other track of this event
#                 pass
#             print 'SET DUT TRACK QUALITY'
            _set_dut_track_quality(tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)

#         print 'TRACK', track_index
#         for dut_index in range(n_duts):
#             print tr_row[track_index][dut_index],
#         print
        # Set number of tracks of last event
        _set_n_tracks(start_index=track_index - n_actual_tracks + 1,
                      stop_index=track_index + 1, tracklets=tracklets,
                      n_actual_tracks=n_actual_tracks,
                      tr_column=tr_column,
                      tr_row=tr_row,
                      min_cluster_distance=min_cluster_distance,
                      n_duts=n_duts)

@njit
def _find_merged_tracks(tracks_array, min_track_distance):  # Check if several tracks are less than min_track_distance apart. Then exclude these tracks (set n_tracks = -1)
    i = 0
    for _ in range(0, tracks_array.shape[0]):
        track_index = i
        if track_index >= tracks_array.shape[0]:
            break
        actual_event = tracks_array[track_index]['event_number']
        for _ in range(track_index, tracks_array.shape[0]):  # Loop over event hits
            if tracks_array[i]['event_number'] != actual_event:  # Next event reached, break loop
                break
            if tracks_array[i]['n_tracks'] < 2:  # Only if the event has more than one track check the min_track_distance
                i += 1
                break
            offset_x, offset_y = tracks_array[i]['offset_0'], tracks_array[i]['offset_1']
            for j in range(i + 1, tracks_array.shape[0]):  # Loop over other event hits
                if tracks_array[j]['event_number'] != actual_event:  # Next event reached, break loop
                    break
                if sqrt((offset_x - tracks_array[j]['offset_0']) * (offset_x - tracks_array[j]['offset_0']) + (offset_y - tracks_array[j]['offset_1']) * (offset_y - tracks_array[j]['offset_1'])) < min_track_distance:
                    tracks_array[i]['n_tracks'] = -1
                    tracks_array[j]['n_tracks'] = -1
            i += 1


def _fit_tracks_loop(track_hits):
    ''' Do 3d line fit and calculate chi2 for each fit. '''
    def line_fit_3d(hits):
        datamean = hits.mean(axis=0)
        offset, slope = datamean, np.linalg.svd(hits - datamean)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
        intersections = offset + slope / slope[2] * (hits.T[2][:, np.newaxis] - offset[2])  # Fitted line and DUT plane intersections (here: points)
        chi2 = np.sum(np.square(hits - intersections), dtype=np.uint32)  # Chi2 of the fit in um
        return datamean, slope, chi2

    slope = np.empty((track_hits.shape[0], 3,))
    offset = np.empty((track_hits.shape[0], 3,))
    chi2 = np.empty((track_hits.shape[0],))

    slope[:] = np.nan
    offset[:] = np.nan
    chi2[:] = np.nan

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        try:
            offset[index], slope[index], chi2[index] = line_fit_3d(actual_hits)
        except np.linalg.linalg.LinAlgError:
            chi2[index] = 1e9

    return offset, slope, chi2


def _function_wrapper_find_tracks_loop(args):  # Needed for multiprocessing call with arguments
    return _find_tracks_loop(*args)
