from __future__ import division

import tempfile
import logging
import numpy as np
import numexpr as ne
import tables as tb
from numba import njit
from scipy.interpolate import splrep, sproot
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import quad

import multiprocessing
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis import analysis_functions
import testbeam_analysis.tools.plot_utils
from testbeam_analysis.cpp import data_struct
from docutils.io import InputError
from gitdb.fun import chunk_size


class SAC(object):
    def __init__(self, table_file_in, table_file_out,
                 func, table_desc={}, table=None, align_at=None,
                 n_cores=None, chunk_size=100000):
        ''' Apply a function to a pytable on multiple cores in chunks.

            Parameters
            ----------
            table_file_in : string
                File name of the file with the table.
            table_file_out : string
                File name with the resulting table.
            func : function
                Function to be applied on table chunks.
            table_desc : dict, None
                Output table parameters from pytables.table().
                If None filters are set from input table and data
                format is set from return value of func.
            table : string, None
                Table name. Needed if multiple tables exists in file.
                If None: only table is used.
            align_at : string, None
                If specified align chunks at this column values
            n_cores : integer, None
                How many cores to use. If None use all available cores.
            chunk_size : int
                Chunk size of the data when reading from file.

            Notes:
            ------
            It follows the split, apply, combine paradigm:
            - split: data is splitted into chunks for multiple processes for
              speed increase
            - map: the function is called on each chunk
            - combine: the results are merged into a result table
            '''

        # Set parameters
        self.table_file_in = table_file_in
        self.n_cores = n_cores
        self.align_at = align_at
        self.func = func
        self.table_desc = table_desc
        self.chunk_size = chunk_size

        # Get the table node name
        with tb.open_file(table_file_in) as in_file:
            if not table:  # Find the table node
                for n in in_file.root:
                    if tb.table.Table == type(n):
                        if not table:
                            node = n
                        else:  # muliple tables
                            raise RuntimeError('No table node defined and'
                                               ' multiple nodes found in file')
                self.node_name = node.name
            else:
                self.node_name = table
                node = in_file.get_node(in_file.root, self.node_name)

            # Set number of rows
            self.n_rows = node.shape[0]

            # Set output parameters for output table
            if 'filter' not in self.table_desc:
                self.table_desc['filters'] = node.filters
            if 'name' not in self.table_desc:
                self.table_desc['name'] = node.name
            if 'title' not in self.table_desc:
                self.table_desc['title'] = node.title

        if not self.n_cores:  # Set n_cores to maximum cores available
            self.n_cores = multiprocessing.cpu_count()

        self._split()
        self._map()

    def _split(self):
        self.start_i, self.stop_i = self._get_split_indeces()

        assert len(self.start_i) == len(self.stop_i)

    def _map(self):
        # Create arguments to call function on multiple cores
        # with changing arguments
        # https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
        multi_args = []
        for i in range(self.n_cores):
            multi_args.append((self.table_file_in,
                               self.node_name,
                               self.func,
                               self.table_desc,
                               self.start_i[i],
                               self.stop_i[i],
                               self.chunk_size))

        print multi_args

        p = multiprocessing.Pool()
        res = p.map(work_wrapper, multi_args)
        print res
        p.close()
        p.join()

    def _get_split_indeces(self):
        ''' Calculates the data for each core.

            Return two lists with start/stop indeces.
            Stop indeces are exclusive.
        '''

        core_chunk_size = self.n_rows // self.n_cores
        start_indeces = range(0, self.n_rows, core_chunk_size)[:-1]

        if not self.align_at:
            stop_indeces = start_indeces[1:]
        else:
            stop_indeces = self._get_next_index(start_indeces)
            start_indeces = [0] + stop_indeces

        stop_indeces.append(self.n_rows)  # Last index always table size
        return start_indeces, stop_indeces

    def _get_next_index(self, indeces):
        ''' Get closest index where the alignment column changes '''

        next_indeces = []
        for index in indeces[1:]:
            with tb.open_file(self.table_file_in) as in_file:
                node = in_file.get_node(in_file.root, self.node_name)
                values = node[index:index + chunk_size][self.align_at]
                value = values[0]
                for i, v in enumerate(values):
                    if v != value:
                        next_indeces.append(index + i)
                        break
                    value = v

        assert len(next_indeces) == self.n_cores - 1
        return next_indeces


def work_wrapper(args):
    return work(*args)


def work(table_file_in, node_name, func,
         table_desc, start_i, stop_i, chunk_size):
    ''' Defines the work per worker.

    Reads data, applies the function and stores data in chunks.

    Has to be outside of class to allow pickling to send it to other process:
    https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-
    when-using-multiprocessing-pool-map
    '''

    with tb.open_file(table_file_in, 'r') as in_file:
        node = in_file.get_node(in_file.root, node_name)

        output_file = tempfile.NamedTemporaryFile(delete=False)
        with tb.open_file(output_file.name, 'w') as out_file:
            # Create result table with specified data format
            if 'description' in table_desc:
                table_out = out_file.create_table(out_file.root,
                                                  **table_desc)
            else:  # Data format unknown
                table_out = None

            for i, data in analysis_utils.data_aligned_at_events(table=node,
                                                                 start_index=start_i,
                                                                 stop_index=stop_i,
                                                                 chunk_size=chunk_size):
                data_ret = func(data)
                # Create table if not existing
                # Extract data type from returned data
                if not table_out:
                    table_out = out_file.create_table(out_file.root,
                                                      description=data.dtype,
                                                      **table_desc)
                table_out.append(data_ret)
    return table_out


if __name__ == '__main__':
    def f(data):
        n_duts = 2
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('x_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('y_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('z_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('charge_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('n_hits_dut_%d' % index, np.int8))
        for dimension in range(3):
            description.append(('offset_%d' % dimension, np.float))
        for dimension in range(3):
            description.append(('slope_%d' % dimension, np.float))
        description.extend([('track_chi2', np.uint32), ('track_quality', np.uint32), ('n_tracks', np.int8)])
        for index in range(n_duts):
            description.append(('xerr_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('yerr_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('zerr_dut_%d' % index, np.float))
        return np.zeros(shape=(10,), dtype=description)

    SAC(table_file_in=r'/home/davidlp/git/testbeam_analysis/testbeam_analysis/examples/data/TestBeamData_FEI4_DUT0.h5',
        table_file_out=r'/home/davidlp/git/testbeam_analysis/testbeam_analysis/examples/data/tets.h5',
        func=f, align_at='event_number')
