''' Implements the often needed split, map, combine paradigm '''
from __future__ import division

import os
import tempfile
import logging
import numpy as np
import tables as tb
from collections import Iterable

from pathos import multiprocessing
from pathos.pools import ProcessPool


class SMC(object):

    def __init__(self, table_file_in, table_file_out,
                 func, table_desc={}, table=None, align_at=None,
                 n_cores=None, chunk_size=1000000):
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
            table : string, iterable of strings, None
                string: Table name. Needed if multiple tables exists in file.
                iterable of strings: possible table names. First existing table
                is used
                None: only table is used independent of name. If multiple
                tables exist exception is raised
            align_at : string, None
                If specified align chunks at this column values
            n_cores : integer, None
                How many cores to use. If None use all available cores.
                If 1 multithreading is disabled, useful for debuging.
            chunk_size : int
                Chunk size of the data when reading from file.

            Notes:
            ------
            It follows the split, apply, combine paradigm:
            - split: data is splitted into chunks for multiple processes for
              speed increase
            - map: the function is called on each chunk. If the chunk per core
              is still too large to fit in memory it is chunked further. The result
              is written to a table per core.
            - combine: the tables are merged into one result table
            '''

        # Set parameters
        self.table_file_in = table_file_in
        self.table_file_out = table_file_out
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
                        else:  # Multiple tables
                            raise RuntimeError('No table node defined and'
                                               ' multiple nodes found in file')
                self.node_name = node.name
            elif isinstance(table, Iterable):  # possible names
                self.node_name = None
                for node_cand in table:
                    try:
                        in_file.get_node(in_file.root, node_cand)
                        self.node_name = node_cand
                    except tb.NoSuchNodeError:
                        pass
                if not self.node_name:
                    raise RuntimeError(
                        'No table nodes with names %s found', str(table))
            else:  # string
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
            # Deactivate multithreading for small data sets
            # Overhead of pools can make multiprocesssing slower
            if self.n_rows < self.chunk_size:
                self.n_cores = 1

        self._split()
        self._map()
        self._combine()

    def _split(self):
        self.start_i, self.stop_i = self._get_split_indeces()
        assert len(self.start_i) == len(self.stop_i)

    def _map(self):
        # Output node name set to input node name
        node_name = self.node_name
        # Try to set output node name if defined
        try:
            node_name = self.table_desc['name']
        except KeyError:
            pass

        if self.n_cores == 1:
            self.tmp_files = [self._work(self.table_file_in,
                                         self.node_name,
                                         self.func,
                                         self.table_desc,
                                         self.start_i[0],
                                         self.stop_i[0],
                                         self.chunk_size)]
        else:
            # Run function in parallel
            # Pathos reuses pools for speed up, is this correct?
            pool = ProcessPool(self.n_cores)
            self.tmp_files = pool.map(self._work,
                                      [self.table_file_in] * self.n_cores,
                                      [self.node_name] * self.n_cores,
                                      [self.func] * self.n_cores,
                                      [self.table_desc] * self.n_cores,
                                      [self.start_i[i]
                                          for i in range(self.n_cores)],
                                      [self.stop_i[i]
                                          for i in range(self.n_cores)],
                                      [self.chunk_size] * self.n_cores)

    def _combine(self):
        # Use first tmp file as result file
        os.rename(self.tmp_files[0], self.table_file_out)
        # Output node name set to input node name
        node_name = self.node_name
        # Try to set output node name if defined
        try:
            node_name = self.table_desc['name']
        except KeyError:
            pass

        with tb.open_file(self.table_file_out, 'r+') as out_file:
            node = out_file.get_node(out_file.root, node_name)
            for f in self.tmp_files[1:]:
                with tb.open_file(f) as in_file:
                    tmp_node = in_file.get_node(in_file.root, node_name)
                    for i in range(0, tmp_node.shape[0], self.chunk_size):
                        node.append(tmp_node[i: i + self.chunk_size])

    def _get_split_indeces(self):
        ''' Calculates the data for each core.

            Return two lists with start/stop indeces.
            Stop indeces are exclusive.
        '''

        core_chunk_size = self.n_rows // self.n_cores
        start_indeces = range(0, self.n_rows, core_chunk_size)

        if not self.align_at:
            stop_indeces = start_indeces[1:]
        else:
            stop_indeces = self._get_next_index(start_indeces)
            start_indeces = [0] + stop_indeces

        stop_indeces.append(self.n_rows)  # Last index always table size

        assert len(stop_indeces) == self.n_cores
        assert len(start_indeces) == self.n_cores
#         raise
        return start_indeces, stop_indeces

    def _get_next_index(self, indeces):
        ''' Get closest index where the alignment column changes '''

        next_indeces = []
        for index in indeces[1:]:
            with tb.open_file(self.table_file_in) as in_file:
                node = in_file.get_node(in_file.root, self.node_name)
                values = node[index:index + self.chunk_size][self.align_at]
                value = values[0]
                for i, v in enumerate(values):
                    if v != value:
                        next_indeces.append(index + i)
                        break
                    value = v

        return next_indeces

    def _work(self, table_file_in, node_name, func,
              table_desc, start_i, stop_i, chunk_size):
        ''' Defines the work per worker.

        Reads data, applies the function and stores data in chunks into a table.
        '''

        # It is needed to import needed modules in every pickled function
        # It is not too clear to what this implies
        import tempfile
        import numpy as np
        import tables as tb
        from testbeam_analysis.tools import analysis_utils

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

                for data, i in self.chunks_aligned_at_events(table=node,
                                                             start_index=start_i,
                                                             stop_index=stop_i,
                                                             chunk_size=chunk_size):

                    data_ret = func(data)
                    # Create table if not existing
                    # Extract data type from returned data
                    if not table_out:
                        table_out = out_file.create_table(out_file.root,
                                                          description=data_ret.dtype,
                                                          **table_desc)
                    table_out.append(data_ret)

        return output_file.name

    def chunks_aligned_at_events(self, table, start_index=None, stop_index=None, chunk_size=10000000):
        '''Takes the table with a event_number column and returns chunks with the size up to chunk_size.
        The chunks are chosen in a way that the events are not splitted.
        Start and the stop indices limiting the table size can be specified to improve performance.
        The event_number column must be sorted.

        Parameters
        ----------
        table : pytables.table
            The data.
        start_index : int
            Start index of data. If None, no limit is set.
        stop_index : int
            Stop index of data. If None, no limit is set.
        chunk_size : int
            Maximum chunk size per read.

        Returns
        -------
        Iterator of tuples
            Data of the actual data chunk and start index for the next chunk.

        Example
        -------
        for data, index in chunk_aligned_at_events(table):
            do_something(data)
            show_progress(index)
        '''

        # Initialize variables
        if not start_index:
            start_index = 0
        if not stop_index:
            stop_index = table.shape[0]

        # Limit max index
        if stop_index > table.shape[0]:
            stop_index = table.shape[0]

        # Special case, one read is enough, data not bigger than one chunk and
        # the indices are known
        if start_index + chunk_size >= stop_index:
            yield table.read(start=start_index, stop=stop_index), stop_index
        else:  # Read data in chunks, chunks do not divide events
            current_start_index = start_index
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size,
                                         stop_index)
                chunk = table[current_start_index:current_stop_index]
                if current_stop_index == stop_index:  # Last chunk
                    yield chunk, stop_index
                    break

                # Find maximum non event number splitting index
                event_numbers = chunk["event_number"]
                last_event = event_numbers[-1]

                # Search for next event number
                chunk_stop_index = np.searchsorted(event_numbers,
                                                   last_event,
                                                   side="left")

                yield chunk[:chunk_stop_index], current_start_index + chunk_stop_index

                current_start_index += chunk_stop_index


if __name__ == '__main__':
    def f(data):
        # It is needed to import needed modules in every pickled function
        # It is not too clear to me what this implies
        import numpy as np
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
        description.extend(
            [('track_chi2', np.uint32), ('track_quality', np.uint32), ('n_tracks', np.int8)])
        for index in range(n_duts):
            description.append(('xerr_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('yerr_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('zerr_dut_%d' % index, np.float))

        a = np.zeros(shape=(data.shape[0],), dtype=description)
        a[:]['event_number'] = data[:]['event_number']
        return a

    SMC(table_file_in=r'../examples/data/TestBeamData_FEI4_DUT0.h5',
        table_file_out=r'tets.h5',
        func=f, align_at='event_number',
        n_cores=1,
        chunk_size=1000)
