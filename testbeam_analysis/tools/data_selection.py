''' Helper functions to select and combine data '''
import numpy as np
import tables as tb
import numexpr as ne
import re
import logging
import progressbar
from numba import njit

from testbeam_analysis.tools import analysis_utils


def combine_hit_files(hit_files, combined_file, event_number_offsets=None, chunk_size=10000000):
    ''' Function to combine hit files of runs with same parameters to increase the statistics.
    Parameters
    ----------
    hit_files : iterable of pytables files
        with hit talbes
    combined_file : pytables file
    event_number_offsets : iterable
        Event numbers at the beginning of each hit file. Needed to synchronize the event number of different DUT files.
    chunk_size : number
        Amount of hits read at once. Limited by available RAM.
    '''

    used_event_number_offsets = []
    with tb.open_file(combined_file, mode="w") as out_file_h5:
        hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=np.dtype([('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)]), title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        event_number_offset = 0
        for index, hit_file in enumerate(hit_files):
            with tb.open_file(hit_file, mode='r') as in_file_h5:
                for hits, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.Hits, chunk_size=chunk_size):
                    hits[:]['event_number'] += event_number_offset
                    hit_table_out.append(hits)
                if not event_number_offsets:
                    event_number_offset += (hits[-1]['event_number'] + 1)
                else:
                    event_number_offset = event_number_offsets[index]
                used_event_number_offsets.append(event_number_offset)

    return used_event_number_offsets


@njit()
def _delete_events(data, fraction):
    result = np.zeros_like(data)
    index_result = 0

    for index in range(data.shape[0]):
        if data[index]['event_number'] % fraction == 0:
            result[index_result] = data[index]
            index_result += 1
    return result[:index_result]


def reduce_hit_files(hit_files, fraction=10, chunk_size=10000000):
    ''' Fast function to delete a fraction of events to allow faster testing of analysis functions.
    Parameters
    ----------
    hit_files : iterable of pytables files
        with hit talbes
    fraction : numer
        The fraction of left over events.
        e.g.: 10 would correspond to n_events = total_events / fraction
    chunk_size : number
        Amount of hits read at once. Limited by available RAM.
    '''

    for hit_file in hit_files:
        with tb.open_file(hit_file, mode='r') as in_file_h5:
            with tb.open_file(hit_file[:-3] + '_reduced.h5', mode="w") as out_file_h5:
                hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=np.dtype([('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)]), title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                for hits, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.Hits, chunk_size=chunk_size):
                    hit_table_out.append(_delete_events(hits, fraction))


def select_hits(hit_file, max_hits=None, condition=None, track_quality=None, track_quality_mask=None, output_file=None, chunk_size=10000000):
    ''' Function to select a fraction of hits fullfilling a condition. Needed for analysis speed up, when
    very large runs are used.
    Parameters
    ----------
    hit_files : iterable of pytables files
        with hit talbes
    max_hits : number
        Number of maximum hits with selection. For data reduction.
        FIXME: is not precise
    condition : string
        A condition that is applied to the hits in numexpr. Only if the expression evaluates to True the hit is taken.
        For example: E.g.: condition = 'track_quality == 2 & event_number < 1000'
    fraction : numer
        The fraction of left over hits where the condiiton applies.
        e.g.: 10 would correspond to n_events = total_events / fraction
    chunk_size : number
        Amount of hits read at once. Limited by available RAM.
    '''

    with tb.open_file(hit_file, mode='r') as in_file_h5:
        if not output_file:
            output_file = hit_file[:-3] + '_reduced.h5'
        with tb.open_file(output_file, mode="w") as out_file_h5:
            for node in in_file_h5.root:
                total_hits = node.shape[0]
                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=total_hits, term_width=80)
                progress_bar.start()
                hit_table_out = out_file_h5.create_table(out_file_h5.root, name=node.name, description=node.dtype, title=node.title, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                for hits, index in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_hits = hits.shape[0]
                    if condition:
                        hits = _select_hits_with_condition(hits, condition)

                    if track_quality:
                        if not track_quality_mask:  # If no mask is defined select all quality bits
                            track_quality_mask = int(0xFFFFFFFF)
                        selection = (hits['track_quality'] & track_quality_mask) == (track_quality)
                        hits = hits[selection]

                    if hits.shape[0] == 0:
                        logging.warning('No hits selected')

                    if max_hits:  # Reduce the number of added hits of this chunk to not exeed max_hits
                        # Calculate number of hits to add for this chunk
                        hit_fraction = max_hits / float(total_hits)  # Fraction of hits to add per chunk
                        selection = np.ceil(np.linspace(0, hits.shape[0], int(hit_fraction * n_hits), endpoint=False)).astype(np.int)
                        selection = selection[selection < hits.shape[0]]
                        hits = hits[selection]

                    hit_table_out.append(hits)
                    progress_bar.update(index)
                progress_bar.finish()


def _select_hits_with_condition(hits_array, condition):
    for variable in set(re.findall(r'[a-zA-Z_]+', condition)):
        exec(variable + ' = hits_array[\'' + variable + '\']')

    return hits_array[ne.evaluate(condition)]

if __name__ == '__main__':
    hit_files = [r'C:\Users\DavidLP\git\testbeam_analysis\examples\data\TrackCandidates_no_align_tmp_final.h5',
                 r'C:\Users\DavidLP\git\testbeam_analysis\examples\data\TestBeamData_FEI4_DUT1.h5']
    select_hits(hit_files[0], max_hits=10000, track_quality=0b001111110011111100111111, condition=None)
