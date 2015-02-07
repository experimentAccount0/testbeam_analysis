"""This script prepares FE-I4 test beam raw data recorded by pyBAR to be analyzed by the simple python test beam analysis.
An installation of pyBAR is required: https://silab-redmine.physik.uni-bonn.de/projects/pybar
- This script does for each DUT in parallel
  - Create a hit tables from the raw data
  - Align the hit table event number to the trigger number to be able to correlate hits in time
  - Rename and select hit info needed for further analysis.
"""

import logging
import numpy as np
import tables as tb
from multiprocessing import Pool

from pybar.analysis import analysis_utils
from pybar.analysis.analyze_raw_data import AnalyzeRawData
from pybar.analysis.RawDataConverter import data_struct

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def analyze_raw_data(input_file):  # FE-I4 raw data analysis
    '''Std. raw data analysis of FE-I4 data. A hit table is ceated for further analysis.

    Parameters
    ----------
    input_file : pytables file
    output_file_hits : pytables file
    '''
    with AnalyzeRawData(raw_data_file=input_file, create_pdf=True) as analyze_raw_data:
        analyze_raw_data.use_trigger_number = False
        analyze_raw_data.interpreter.use_tdc_word(False)
        analyze_raw_data.create_hit_table = True
        analyze_raw_data.create_meta_event_index = True
        analyze_raw_data.create_trigger_error_hist = True
        analyze_raw_data.create_rel_bcid_hist = True
        analyze_raw_data.create_error_hist = True
        analyze_raw_data.create_service_record_hist = True
        analyze_raw_data.create_occupancy_hist = False
        analyze_raw_data.create_tot_hist = False
        analyze_raw_data.n_bcid = 16
        analyze_raw_data.n_injections = 100
        analyze_raw_data.max_tot_value = 13
        analyze_raw_data.interpreter.set_debug_output(False)
        analyze_raw_data.interpreter.set_info_output(False)
        analyze_raw_data.interpreter.set_warning_output(False)
        analyze_raw_data.clusterizer.set_warning_output(False)
        analyze_raw_data.interpret_word_table()
        analyze_raw_data.interpreter.print_summary()
        analyze_raw_data.plot_histograms()


def process_dut(raw_data_file):
    ''' Process and formate the raw data.'''
    analyze_raw_data(raw_data_file)
    align_events(raw_data_file[:-3] + '_interpreted.h5', raw_data_file[:-3] + '_event_aligned.h5')
    format_hit_table(raw_data_file[:-3] + '_event_aligned.h5', raw_data_file[:-3] + '_aligned.h5')


def align_events(input_file, output_file, chunk_size=10000000):
    ''' Selects only hits from good events and checks the distance between event number and trigger number for each hit.
    If the FE data allowed a successfull event recognizion the distance is always constant (besides the fact that the trigger number overflows).
    Otherwise the event number is corrected by the trigger number. How often an inconstistency occurs is counted as well as the number of events that had to be corrected.
    Remark: Only one event analyzed wrong shifts all event numbers leading to no correlation! But usually data does not have to be corrected.

    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    chunk_size :  int
        How many events are read at once into RAM for correction.
    '''
    logging.info('Align events to trigger number in %s' % input_file)

    with tb.open_file(input_file, 'r') as in_file_h5:
        hit_table = in_file_h5.root.Hits
        jumps = []  # variable to determine the jumps in the event-number to trigger-number offset
        n_fixed_events = 0  # events that were fixed
        with tb.open_file(output_file, 'w') as out_file_h5:
            hit_table_description = data_struct.HitInfoTable().columns.copy()
            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=hit_table_description, title='Selected hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False), chunkshape=(chunk_size,))
            # Correct hit event number
            for hits, _ in analysis_utils.data_aligned_at_events(hit_table, chunk_size=chunk_size):
                selected_hits = hits[(hits['event_status'] & 0b0000011111111111) == 0b0000000000000000]  # no error at all
                selector = np.array((np.mod(selected_hits['event_number'], 32768) - selected_hits['trigger_number']), dtype=np.int32)
                jumps.extend(np.unique(selector).tolist())
                n_fixed_events += np.count_nonzero(selector)
                selected_hits['event_number'] = np.divide(selected_hits['event_number'], 32768) * 32768 + selected_hits['trigger_number']
                hit_table_out.append(selected_hits)

        jumps = np.unique(np.array(jumps))
        logging.info('Found %d inconsistencies in the event number. %d events had to be corrected.' % (jumps[jumps != 0].shape[0], n_fixed_events))


def format_hit_table(input_file, output_file):
    ''' Selects and renames important columns for test beam analysis and stores them into a new file.

    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    '''

    with tb.open_file(input_file, 'r') as in_file_h5:
        hits = in_file_h5.root.Hits[:]
        hits_formatted = np.zeros((hits.shape[0], ), dtype=[('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])
        with tb.open_file(output_file, 'w') as out_file_h5:
            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=hits_formatted.dtype, title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hits_formatted['event_number'] = hits['event_number']
            hits_formatted['frame'] = hits['relative_BCID']
            hits_formatted['column'] = hits['column']
            hits_formatted['row'] = hits['row']
            hits_formatted['charge'] = hits['tot']
            hit_table_out.append(hits_formatted)

if __name__ == "__main__":
    # Input raw data file names
    raw_data_files = ['C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_132_SCC_29_3.4_GeV_0.h5',  # the first DUT is the master reference DUT
                      'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_213_SCC_99_3.4_GeV_0.h5',
                      'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_214_SCC_146_3.4_GeV_0.h5',
                      'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_201_SCC_166_3.4_GeV_0.h5',
                      'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_207_SCC_112_3.4_GeV_0.h5',
                      'C:\\Users\\DavidLP\\Desktop\\tb\\BOARD_ID_216_SCC_45_3.4_GeV_0.h5']  # the last DUT is the second reference DUT

    # Do seperate DUT data processing in parallel. The output is a formatted hit table.
    pool = Pool()
    pool.map(process_dut, raw_data_files)
