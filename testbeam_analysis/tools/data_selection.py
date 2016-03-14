''' Helper functions to select and combine data '''
import numpy as np
import tables as tb
from numba import njit

from testbeam_analysis import analysis_utils


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
        hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=np.dtype([('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)]), title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
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
    ''' Function to delete a fraction of events to allow faster testing of analysis functions.s
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
                hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=np.dtype([('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)]), title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                for hits, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.Hits, chunk_size=chunk_size):
                    hit_table_out.append(_delete_events(hits, fraction))

if __name__ == '__main__':
    hit_files = [r'C:\Users\DavidLP\git\testbeam_analysis\examples\data\TestBeamData_FEI4_DUT0.h5',
                 r'C:\Users\DavidLP\git\testbeam_analysis\examples\data\TestBeamData_FEI4_DUT1.h5']
    reduce_hit_files(hit_files)
