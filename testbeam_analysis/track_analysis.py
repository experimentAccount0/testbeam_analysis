''' Track finding and fitting functions are listed here.'''

import logging
import progressbar
import tables as tb
import numpy as np
import itertools

from math import sqrt
from numba import njit
from multiprocessing import Pool, cpu_count
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis import plot_utils


def find_tracks(tracklets_file, alignment_file, track_candidates_file, limit_events=None):
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
    logging.info('=== Build tracks from tracklets ===')

    # Get alignment errors from file
    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] // 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    with tb.open_file(tracklets_file, mode='r') as in_file_h5:
        with tb.open_file(track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=in_file_h5.root.Tracklets.dtype, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            n_duts = sum(['column' in col for col in in_file_h5.root.Tracklets.dtype.names])
            print "n_duts", n_duts

            print "read max event number"
            max_event_number = in_file_h5.root.Tracklets.cols.event_number[-1]
            print max_event_number
            tracklets_file_index = 0
            tracklets_data_chunk = None
            for event_number in itertools.count(0, 100000):
                if event_number > max_event_number:
                    break
                # TODO: fix
                if limit_events and total_events >= limit_events:
                    break
                last_event_number = event_number + 100000 - 1
                print event_number, last_event_number
                print tracklets_file_index

                # Prepare data for track finding, create arrays for column, row and charge data
                tracklets_data_chunk = in_file_h5.root.Tracklets.read_where('(event_number <= %s)' % last_event_number, start=tracklets_file_index, stop=in_file_h5.root.Tracklets.nrows).view(np.recarray)
                tr_column = tracklets_data_chunk['column_dut_0']
                tr_row = tracklets_data_chunk['row_dut_0']
                tr_charge = tracklets_data_chunk['charge_dut_0']
                for dut_index in range(n_duts - 1):
                    tr_column = np.vstack((tr_column, tracklets_data_chunk['column_dut_%d' % (dut_index + 1)]))
                    tr_row = np.vstack((tr_row, tracklets_data_chunk['row_dut_%d' % (dut_index + 1)]))
                    tr_charge = np.vstack((tr_charge, tracklets_data_chunk['charge_dut_%d' % (dut_index + 1)]))
                tr_column = np.transpose(tr_column)
                tr_row = np.transpose(tr_row)
                tr_charge = np.transpose(tr_charge)

                # Perform the track finding with jitted loop
                tracklets_data_chunk, tr_column, tr_row, tr_charge = _find_tracks_loop(tracklets_data_chunk, tr_column, tr_row, tr_charge, column_sigma, row_sigma)

                # Merge result data from arrays into one recarray
                combined = np.column_stack((tracklets_data_chunk.event_number, tr_column, tr_row, tr_charge, tracklets_data_chunk.track_quality, tracklets_data_chunk.n_tracks))
                combined = np.core.records.fromarrays(combined.transpose(), dtype=tracklets_data_chunk.dtype)

                track_candidates.append(combined)


def find_tracks_corr(tracklets_file, alignment_file, track_candidates_file, pixel_size):
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
    logging.info('=== Build tracks from TrackCandidates ===')

    # Get alignment errors from file
    with tb.open_file(alignment_file, mode='r') as in_file_h5:
        correlations = in_file_h5.root.Alignment[:]
        column_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        row_sigma = np.zeros(shape=(correlations.shape[0] / 2) + 1)
        column_sigma[0], row_sigma[0] = 0, 0  # DUT0 has no correlation error
        for index in range(1, correlations.shape[0] // 2 + 1):
            column_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][0]]
            row_sigma[index] = correlations['sigma'][np.where(correlations['dut_x'] == index)[0][1]]

    with tb.open_file(tracklets_file, mode='r') as in_file_h5:
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

        with tb.open_file(track_candidates_file, mode='w') as out_file_h5:
            track_candidates2 = out_file_h5.create_table(out_file_h5.root, name='TrackCandidates', description=in_file_h5.root.TrackCandidates.description, title='Track candidates', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            track_candidates2.append(combined)


def optimize_track_alignment(trackcandidates_file, alignment_file, fraction=1, correlated_only=False):
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
        Use only every fraction-th hit for the alignment correction. For speed up. 1 means all hits are used
    correlated_only : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file
    '''
    logging.info('=== Optimize track alignment ===')
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
    trackcandidates_file : string
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


def fit_tracks(track_candidates_file, tracks_file, z_positions, fit_duts=None, ignore_duts=None, include_duts=[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], track_quality=1, max_tracks=None, output_pdf=None, use_correlated=False):
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

    with PdfPages(output_pdf) as output_fig:
        with tb.open_file(track_candidates_file, mode='r') as in_file_h5:
            with tb.open_file(tracks_file, mode='w') as out_file_h5:
                track_candidates = in_file_h5.root.TrackCandidates[:]
                n_duts = sum(['column' in col for col in in_file_h5.root.TrackCandidates.dtype.names])
                fit_duts = fit_duts if fit_duts else range(n_duts)
                for fit_dut in fit_duts:  # Loop over the DUTs where tracks shall be fitted for
                    logging.info('Fit tracks for DUT %d', fit_dut)

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
                    good_track_selection = (track_candidates['track_quality'] & (dut_selection << (track_quality * 8))) == (dut_selection << (track_quality * 8))
                    if max_tracks:  # Option to neglect events with too many hits
                        good_track_selection = np.logical_and(good_track_selection, track_candidates['n_tracks'] <= max_tracks)

                    logging.info('Lost %d tracks due to track quality cuts, %d percent ', good_track_selection.shape[0] - np.count_nonzero(good_track_selection), (1. - float(np.count_nonzero(good_track_selection) / float(good_track_selection.shape[0]))) * 100.)

                    if use_correlated:  # Reduce track selection to correlated DUTs only
                        good_track_selection &= (track_candidates['track_quality'] & (quality_mask << 24) == (quality_mask << 24))
                        logging.info('Lost due to correlated cuts %d', good_track_selection.shape[0] - np.sum(track_candidates['track_quality'] & (quality_mask << 24) == (quality_mask << 24)))

                    good_track_candidates = track_candidates[good_track_selection]

                    # Prepare track hits array to be fitted
                    n_fit_duts = bin(dut_selection).count("1")
                    index, n_tracks = 0, good_track_candidates['event_number'].shape[0]  # Index of tmp track hits array
                    track_hits = np.zeros((n_tracks, n_fit_duts, 3))
                    for dut_index in range(0, n_duts):  # Fill index loop of new array
                        if (1 << dut_index) & dut_selection == (1 << dut_index):  # True if DUT is used in fit
                            xyz = np.column_stack((good_track_candidates['column_dut_%s' % dut_index], good_track_candidates['row_dut_%s' % dut_index], np.repeat(z_positions[dut_index], n_tracks)))
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
                    offsets = np.concatenate([i[0] for i in results])  # merge offsets from all cores in results
                    slopes = np.concatenate([i[1] for i in results])  # merge slopes from all cores in results
                    chi2s = np.concatenate([i[2] for i in results])  # merge chi2 from all cores in results
                    tracks_array = create_results_array(good_track_candidates, slopes, offsets, chi2s, n_duts)
                    tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    tracklets_table.append(tracks_array)

                    # Plot chi2 distribution
                    plot_utils.plot_track_chi2(chi2s, fit_dut, output_fig)


# Helper functions that are not meant to be called during analysis
@njit
def _set_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma):
    # Set track quality of actual DUT from closest DUT hit, if hit is within 2 or 5 sigma range
    # Quality 0 (there is one hit, no matter of sigma distance) is already set
    column, row = tr_column[track_index][dut_index], tr_row[track_index][dut_index]
    if row != 0:  # row = 0 is no hit
        column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
        if column_distance < 2 * actual_column_sigma and row_distance < 2 * actual_row_sigma:  # High quality track hits
            actual_track.track_quality |= (65793 << dut_index)
        elif column_distance < 5 * actual_column_sigma and row_distance < 5 * actual_row_sigma:  # Low quality track hits
            actual_track.track_quality |= (257 << dut_index)
    else:
        actual_track.track_quality &= (~ (65793 << dut_index))  # Unset track quality


@njit
def _swap_hits(tracklets, tr_column, tr_row, tr_charge, track_index, dut_index, hit_index, column, row, charge):
    tmp_column, tmp_row, tmp_charge = tr_column[track_index][dut_index], tr_row[track_index][dut_index], tr_charge[track_index][dut_index]
    tr_column[track_index][dut_index], tr_row[track_index][dut_index], tr_charge[track_index][dut_index] = column, row, charge
    tr_column[hit_index][dut_index], tr_row[hit_index][dut_index], tr_charge[hit_index][dut_index] = tmp_column, tmp_row, tmp_charge


@njit
def _find_tracks_loop(tracklets, tr_column, tr_row, tr_charge, column_sigma, row_sigma):
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
    best_hit_distance = 0.

    for track_index, actual_track in enumerate(tracklets):  # Loop over all possible tracks
        # Set variables for new event
        if actual_track.event_number != actual_event_number:  # Detect new event
            actual_event_number = actual_track.event_number
            for i in range(n_actual_tracks):  # Set number of tracks of previous event
                tracklets[track_index - 1 - i].n_tracks = n_actual_tracks
            n_actual_tracks = 0
            actual_hit_track_index = track_index

        n_actual_tracks += 1
        first_hit_set = False

        for dut_index in range(n_duts):  # loop over all DUTs in the actual track

            actual_column_sigma, actual_row_sigma = column_sigma[dut_index], row_sigma[dut_index]

            if not first_hit_set and tr_row[track_index][dut_index] != 0:  # search for first DUT that registered a hit (row != 0)
                actual_track_column, actual_track_row = tr_column[track_index][dut_index], tr_row[track_index][dut_index]
                first_hit_set = True
                tracklets[track_index].track_quality |= (65793 << dut_index)  # First track hit has best quality by definition
            else:  # Find best (closest) DUT hit
                close_hit_found = False
                for hit_index in range(actual_hit_track_index, tracklets.shape[0]):  # Loop over all not sorted hits of actual DUT
                    if tracklets[hit_index].event_number != actual_event_number:
                        break
                    column, row, charge, quality = tr_column[hit_index][dut_index], tr_row[hit_index][dut_index], tr_charge[hit_index][dut_index], tracklets[hit_index].track_quality
                    column_distance, row_distance = abs(column - actual_track_column), abs(row - actual_track_row)
                    hit_distance = sqrt(column_distance * column_distance + row_distance * row_distance)

                    if row != 0:  # Track hit found
                        actual_track.track_quality |= (1 << dut_index)  # track quality 0 for DUT dut_index (in first byte one bit set)
                        quality |= (1 << dut_index)

                    if row != 0 and not close_hit_found and column_distance < 5. * actual_column_sigma and row_distance < 5. * actual_row_sigma:  # good track hit (5 sigma search region)
                        if tracklets[hit_index].track_quality & (65793 << dut_index) == (65793 << dut_index):  # Check if hit is already a close hit, then do not move
                            _set_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)
                            continue
                        if tracklets[hit_index].track_quality & (257 << dut_index) == (257 << dut_index):  # Check if old hit is closer, then do not move
                            column_distance_old, row_distance_old = abs(column - tr_column[hit_index][0]), abs(row - tr_row[hit_index][0])
                            hit_distance_old = sqrt(column_distance_old * column_distance_old + row_distance_old * row_distance_old)
                            if hit_distance > hit_distance_old:  # Only take hit if it fits better to actual track
                                _set_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)
                                continue
                        _swap_hits(tracklets, tr_column, tr_row, tr_charge, track_index, dut_index, hit_index, column, row, charge)
                        best_hit_distance = hit_distance
                        close_hit_found = True
                    elif row != 0 and close_hit_found and hit_distance < best_hit_distance:  # found better track hit
                        _swap_hits(tracklets, tr_column, tr_row, tr_charge, track_index, dut_index, hit_index, column, row, charge)
                        best_hit_distance = hit_distance

                    _set_track_quality(tracklets, tr_column, tr_row, track_index, dut_index, actual_track, actual_track_column, actual_track_row, actual_column_sigma, actual_row_sigma)

        # Set number of tracks of last event
        for i in range(n_actual_tracks):
            tracklets[track_index - i].n_tracks = n_actual_tracks

    return tracklets, tr_column, tr_row, tr_charge


def _fit_tracks_loop(track_hits):
    ''' Do 3d line fit and calculate chi2 for each fit. '''
    def line_fit_3d(hits):
        datamean = hits.mean(axis=0)
        offset, slope = datamean, np.linalg.svd(hits - datamean)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
        intersections = offset + slope / slope[2] * (hits.T[2][:, np.newaxis] - offset[2])  # Fitted line and DUT plane intersections (here: points)
        chi2 = np.sum(np.square(hits - intersections), dtype=np.uint32)  # Chi2 of the fit in um
        return datamean, slope, chi2

    slope = np.zeros((track_hits.shape[0], 3, ))
    offset = np.zeros((track_hits.shape[0], 3, ))
    chi2 = np.zeros((track_hits.shape[0], ))

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        try:
            offset[index], slope[index], chi2[index] = line_fit_3d(actual_hits)
        except np.linalg.linalg.LinAlgError:
            chi2[index] = 1e9

    return offset, slope, chi2


def _function_wrapper_find_tracks_loop(args):  # Needed for multiprocessing call with arguments
    return _find_tracks_loop(*args)
