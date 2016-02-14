''' Helper functions to select and combine data '''
import numpy as np
import tables as tb
import progressbar

from testbeam_analysis import analysis_utils


def combine_hit_files(hit_files, combined_file, chunk_size=10000000):
    ''' Function to combine hit files of runs with same parameters to increase the statistics.
    Parameters
    ----------
    hit_files : iterable of pytables files
    combined_file : pytables file
    '''
    with tb.open_file(combined_file, mode="w") as out_file_h5:
        hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=np.dtype([('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)]), title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        event_number_offset = 0
        for hit_file in hit_files:
            with tb.open_file(hit_file, mode='r') as in_file_h5:
                for hits, _ in analysis_utils.data_aligned_at_events(in_file_h5.root.Hits, chunk_size=chunk_size):
                    hits[:]['event_number'] += event_number_offset
                    hit_table_out.append(hits)
                event_number_offset += (hits[-1]['event_number'] + 1)

if __name__ =='__main__':
    hit_files=[r'C:\Users\DavidLP\Desktop\TB\RUN_0\Analysed_Data\14_scc_167_ext_trigger_scan_aligned.h5',
               r'C:\Users\DavidLP\Desktop\TB\RUN_2\Analyzed_Data\16_scc_167_ext_trigger_scan_aligned.h5']
    combined_file = r'C:\Users\DavidLP\Desktop\TB\RUN_2\Analyzed_Data\16_scc_167_ext_trigger_scan_aligned_TEST.h5'
    combine_hit_files(hit_files, combined_file)