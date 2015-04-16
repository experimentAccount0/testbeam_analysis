''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''

import unittest
import os
import tables as tb
import numpy as np
import progressbar

from pyTestbeamAnalysis.clusterizer import data_struct
from pyTestbeamAnalysis import analysis_utils
from pyTestbeamAnalysis import analysis_functions
from _sqlite3 import Row

tests_data_folder = 'tests//test_analysis//'


def get_random_data(n_hits, seed=0):
    np.random.seed(seed)
    event_numbers = np.arange(n_hits, dtype=np.int64).repeat(2)[:n_hits]
    ref_column, ref_row = np.random.uniform(high=80, size=n_hits), np.random.uniform(high=336, size=n_hits)
    column, row = ref_column.copy(), ref_row.copy()
    corr = np.ascontiguousarray(np.ones(shape=event_numbers.shape, dtype=np.uint8))  # array to signal correlation to be ables to omit not correlated events in the analysis

    event_numbers = np.ascontiguousarray(event_numbers)
    ref_column = np.ascontiguousarray(ref_column)
    column = np.ascontiguousarray(column)
    ref_row = np.ascontiguousarray(ref_row)
    row = np.ascontiguousarray(row)

    return event_numbers, ref_column, column, ref_row, row, corr


def get_array_differences(first_array, second_array):
    '''Takes two numpy.ndarrays and compares them on a column basis.
    Different column data types, missing columns and columns with different values are returned in a string.

    Parameters
    ----------
    first_array : numpy.ndarray
    second_array : numpy.ndarray

    Returns
    -------
    string
    '''
    if first_array.dtype.names is None:  # normal nd.array
        return ': Sum first array: ' + str(np.sum(first_array)) + ', Sum second array: ' + str(np.sum(second_array))
    else:
        return_str = ''
        for column_name in first_array.dtype.names:
            first_column = first_array[column_name]
            try:
                second_column = second_array[column_name]
            except ValueError:
                return_str += 'No ' + column_name + ' column found. '
                continue
            if (first_column.dtype != second_column.dtype):
                return_str += 'Column ' + column_name + ' has different data type. '
            try:
                if not (first_column == second_column).all():  # check if the data of the column is equal
                    return_str += 'Column ' + column_name + ' not equal. '
            except AttributeError:
                if not (first_column == second_column):
                    return_str += 'Column ' + column_name + ' not equal. '
        for column_name in second_array.dtype.names:
            try:
                first_array[column_name]
            except ValueError:
                return_str += 'Additional column ' + column_name + ' found. '
                continue
        return ': ' + return_str


def compare_h5_files(first_file, second_file, expected_nodes=None, detailed_comparison=True):
    '''Takes two hdf5 files and check for equality of all nodes.
    Returns true if the node data is equal and the number of nodes is the number of expected nodes.
    It also returns a error string containing the names of the nodes that are not equal.

    Parameters
    ----------
    first_file : string
        Path to the first file.
    second_file : string
        Path to the first file.
    expected_nodes : Int
        The number of nodes expected in the second_file. If not specified the number of nodes expected in the second_file equals
        the number of nodes in the first file.

    Returns
    -------
    bool, string
    '''
    checks_passed = True
    error_msg = ""
    with tb.open_file(first_file, 'r') as first_h5_file:
        with tb.open_file(second_file, 'r') as second_h5_file:
            expected_nodes = sum(1 for _ in enumerate(first_h5_file.root)) if expected_nodes is None else expected_nodes  # set the number of expected nodes
            nodes = sum(1 for _ in enumerate(second_h5_file.root))  # calculated the number of nodes
            if nodes != expected_nodes:
                checks_passed = False
                error_msg += 'The number of nodes in the file is wrong.\n'
            for node in second_h5_file.root:  # loop over all nodes and compare each node, do not abort if one node is wrong
                node_name = node.name
                try:
                    expected_data = first_h5_file.get_node(first_h5_file.root, node_name)[:]
                    data = second_h5_file.get_node(second_h5_file.root, node_name)[:]
                    try:
                        if not (expected_data == data).all():  # compare the arrays for each element
                            checks_passed = False
                            error_msg += node_name
                            if detailed_comparison:
                                error_msg += get_array_differences(expected_data, data)
                            error_msg += '\n'
                    except AttributeError:  # .all() only works on non scalars, recarray is somewhat a scalar
                        if not (expected_data == data):
                            checks_passed = False
                            error_msg += node_name
                            if detailed_comparison:
                                error_msg += get_array_differences(expected_data, data)
                            error_msg += '\n'
                except tb.NoSuchNodeError:
                    checks_passed = False
                    error_msg += 'Unknown node ' + node_name + '\n'
    return checks_passed, error_msg


class TestAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_fix_event_alignment(self):  # check with difficult data
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(50)
        column, row = np.zeros_like(column), np.zeros_like(row)
        # Create not correlated events
        column[10:] = ref_column[0:-10]
        row[10:] = ref_row[0:-10]
        column[20:] = ref_column[20:]
        row[20:] = ref_row[20:]
        row[10] = 43.9051

        # Create no hits (virtual hits) in DUT 1
        column[5:15] = 0
        row[5:15] = 0

        print event_numbers.dtype

        tmp = np.delete(np.column_stack((event_numbers, ref_row, row, corr)), [5, 15, 21, 29], 0)
        event_numbers, ref_row, row, corr = tmp[:, 0], tmp[:, 1], tmp[:, 2], tmp[:, 3]

        event_numbers = event_numbers.astype(int)

        print event_numbers.dtype

        for index, (event, hit_1, hit_2, c) in enumerate(np.column_stack((event_numbers, ref_row, row, corr))):
            print index, int(event), hit_1, hit_2, c
        print '___________________'

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        for index, (event, hit_1, hit_2, c) in enumerate(np.column_stack((event_numbers, ref_row, row, corr))):
            print index, int(event), hit_1, hit_2, c
        print '___________________'

        # Check fixes counter
        self.assertEqual(n_fixes, 2)

        # Correlation flag check
        self.assertEqual(corr[0], 0)
        self.assertTrue(np.all(corr[0:4] == 0))
        self.assertTrue(np.all(corr[5:8] == 1))
        self.assertTrue(np.all(corr[9:17] == 0))
        self.assertTrue(np.all(corr[18:] == 1))

        # The data is correlated here
        self.assertTrue(np.all(ref_column[5:8] == column[5:8]))
        self.assertTrue(np.all(ref_row[5:8] == row[5:8]))
        self.assertTrue(np.all(ref_column[18:] == column[18:]))
        self.assertTrue(np.all(ref_row[18:] == row[18:]))

        # Shifted data has to leave zeroes
        self.assertEqual(column[0], 0)
        self.assertEqual(row[0], 0)
        self.assertTrue(np.all(row[0:4] == 0))
        self.assertTrue(np.all(column[0:4] == 0))
        self.assertTrue(np.all(row[9:17] == 0))
        self.assertTrue(np.all(column[9:17] == 0))

    def test_missing_data(self):  # check behavior with missing data, but correlation
        event_numbers, ref_column, column, ref_row, row, _ = get_random_data(50)

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=10)

        # Check that no fixes where done
        self.assertEqual(n_fixes, 0)

        # Correlation flag check
        self.assertTrue(np.all(corr[0:4] == 0))
        self.assertTrue(np.all(corr[5:8] == 1))
        self.assertTrue(np.all(corr[9:17] == 0))
        self.assertTrue(np.all(corr[18:] == 0))

        # Data is the same where there are hits and correlation flag is set
        self.assertTrue(np.all(ref_column[np.logical_and(corr == 1, column != 0)] == column[np.logical_and(corr == 1, column != 0)]))
        self.assertTrue(np.all(row[np.logical_and(corr == 1, column != 0)] == ref_row[np.logical_and(corr == 1, column != 0)]))

#     def test_correlation_flag(self):  # check behavior of the correlation flag
#         event_numbers, ref_column, column, ref_row, row, corr = get_random_data(500)
#         column[5:16] = 0
#         row[5:16] = 0
#         column[16:19] = ref_column[6:9]
#         row[16:19] = ref_row[6:9]
#         corr[17] = 0  # create not correlated event
# 
#         # Check with correlation hole
#         n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=10)
# 
#         self.assertEqual(n_fixes, 0)  # no fixes are expected
#         # Correlation flag check
#         self.assertTrue(np.all(corr[0:6] == 1))
#         self.assertTrue(np.all(corr[6:19] == 0))
#         self.assertTrue(np.all(corr[19:] == 1))
# 
#         event_numbers, ref_column, column, ref_row, row, corr = get_random_data(50)
#         column[5:16] = 0
#         row[5:16] = 0
#         column[16:19] = ref_column[6:9]
#         row[16:19] = ref_row[6:9]
#         corr[17] = 0  # create not correlated event
# 
#         # check with event copying
#         n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=1, correlation_search_range=100, good_events_search_range=10)
# 
#         self.assertEqual(n_fixes, 2)  # 2 fixes are expected
# 
#         # Correlation flag check
#         self.assertTrue(np.all(corr[:7] == 1))
#         self.assertEqual(corr[7], 0)
#         self.assertTrue(np.all(corr[8:10]))
#         self.assertTrue(np.all(corr[10:19] == 0))
#         self.assertTrue(np.all(corr[19:] == 1))


if __name__ == '__main__':
    tests_data_folder = 'test_analysis//'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
