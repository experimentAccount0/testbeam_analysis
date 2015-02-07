"""This script prepares test beam data recorded by eudaq to be analyzed by the simple python test beam analysis.

"""

import logging
import numpy as np
import tables as tb
from multiprocessing import Pool

from pybar.analysis import analysis_utils
from pybar.analysis.analyze_raw_data import AnalyzeRawData
from pybar.analysis.RawDataConverter import data_struct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def process_dut(raw_data_file):
    ''' Process and formate the raw data.'''
    align_events(raw_data_file, raw_data_file[:-3] + '_event_aligned.h5')
    format_hit_table(raw_data_file)


def align_events(input_file, output_file, chunk_size=10000000):
    '''
    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    chunk_size :  int
        How many events are read at once into RAM for correction.
    '''
    logging.warning('Aligning events and correcting timestamp / tlu trigger number is not implemented. We trust the EUDAQ event building now.')


def format_hit_table(input_file):
    ''' Selects and renames important columns for test beam analysis and stores them into a new file.

    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    '''

    with tb.open_file(input_file, 'r') as in_file_h5:
        min_timestamp = min([node[0]['timestamp'] for node in in_file_h5.root])
        for node in in_file_h5.root:
            hits = node[:]
            for dut_index in np.unique(hits['plane']):
                with tb.open_file(input_file[:-3] + '_DUT%d.h5' % dut_index, 'w') as out_file_h5:
                    hits_actual_dut = hits[hits['plane'] == dut_index]
                    hits_formatted = np.zeros((hits_actual_dut.shape[0], ), dtype=[('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])
                    hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=hits_formatted.dtype, title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    hits_formatted['event_number'] = hits_actual_dut['timestamp'] - min_timestamp
                    hits_formatted['frame'] = hits_actual_dut['frame']
                    hits_formatted['column'] = hits_actual_dut['x'] + 1
                    hits_formatted['row'] = hits_actual_dut['y'] + 1
                    hits_formatted['charge'] = hits_actual_dut['val']
                    hit_table_out.append(hits_formatted)

if __name__ == "__main__":
    n_duts = 7  # the number of used DUTs (= planes)
    # Input raw data file names
    raw_data_files = ['D:\\TestBeamDataElsa_149.h5', ]#,  # the first DUT is the master reference DUT
#                       'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_213_SCC_99_3.4_GeV_0.h5',
#                       'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_214_SCC_146_3.4_GeV_0.h5',
#                       'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_201_SCC_166_3.4_GeV_0.h5',
#                       'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_207_SCC_112_3.4_GeV_0.h5',
#                       'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_216_SCC_45_3.4_GeV_0.h5']  # the last DUT is the second reference DUT

    # Do seperate DUT data processing in parallel. The output is a formatted hit table.
#     pool = Pool()
#     pool.map(process_dut, raw_data_files)
    process_dut(raw_data_files[0])
