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
from nose import SkipTest

tests_data_folder = 'tests//test_analysis//'


def get_random_data(n_hits, hits_per_event=2, seed=0):
    np.random.seed(seed)
    event_numbers = np.arange(n_hits, dtype=np.int64).repeat(hits_per_event)[:n_hits]
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

    def test_fix_event_alignment(self):  # check with multiple jumps data
        event_numbers, ref_column, column, ref_row, row, _ = get_random_data(50)
        column, row = np.zeros_like(column), np.zeros_like(row)
        # Create not correlated events
        column[10:] = ref_column[0:-10]
        row[10:] = ref_row[0:-10]
        column[20:] = ref_column[20:]
        row[20:] = ref_row[20:]
        row[10] = 3.14159
        column[10] = 3.14159

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        # Check fixes counter
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[0:10] == 1))
        self.assertTrue(np.all(corr[20:] == 1))

        # The data is correlated here
        self.assertTrue(np.all(ref_column[1:10] == column[1:10]))
        self.assertTrue(np.all(ref_row[1:10] == row[1:10]))
        self.assertTrue(np.all(ref_column[20:] == column[20:]))
        self.assertTrue(np.all(ref_row[20:] == row[20:]))

        # Shifted data has to leave zeroes
        self.assertEqual(column[0], 3.14159)
        self.assertEqual(row[0], 3.14159)
        self.assertTrue(np.all(row[10:20] == 0))
        self.assertTrue(np.all(column[10:20] == 0))

    def test_missing_data(self):  # check behavior with missing data, but correlation
        event_numbers, ref_column, column, ref_row, row, _ = get_random_data(50)

        # Create no hits (virtual hits) in DUT 1
        column[5:15] = 0
        row[5:15] = 0

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        # Check that no fixes where done
        self.assertEqual(n_fixes, 0)

        # Correlation flag check
        self.assertTrue(np.all(corr[:6] == 1))
        self.assertTrue(np.all(corr[6:14] == 0))
        self.assertTrue(np.all(corr[14:] == 1))

        # Data is the same where there are hits and correlation flag is set
        self.assertTrue(np.all(ref_column[np.logical_and(corr == 1, column != 0)] == column[np.logical_and(corr == 1, column != 0)]))
        self.assertTrue(np.all(row[np.logical_and(corr == 1, column != 0)] == ref_row[np.logical_and(corr == 1, column != 0)]))

    def test_correlation_flag(self):  # check behavior of the correlation flag
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(500)
        column[5:16] = 0
        row[5:16] = 0
        column[16:20] = ref_column[6:10]
        row[16:20] = ref_row[6:10]
        corr[16:18] = 0  # create not correlated event

        # Check with correlation hole
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        self.assertEqual(n_fixes, 0)  # no fixes are expected
        # Correlation flag check
        self.assertTrue(np.all(corr[0:6] == 1))
        self.assertTrue(np.all(corr[6:19] == 0))
        self.assertTrue(np.all(corr[20:] == 1))

        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(50)
        column[5:16] = 0
        row[5:16] = 0
        column[16:20] = ref_column[6:10]
        row[16:20] = ref_row[6:10]
        corr[16:18] = 0  # create not correlated event

        # check with event copying
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=1, correlation_search_range=100, good_events_search_range=4)

        self.assertEqual(n_fixes, 1)  # 1 fixe are expected

        # Correlation flag check
        self.assertTrue(np.all(corr[0:6] == 1))
        self.assertTrue(np.all(corr[6:8] == 0))
        self.assertTrue(np.all(corr[8:10] == 1))
        self.assertTrue(np.all(corr[10:20] == 0))
        self.assertTrue(np.all(corr[20:] == 1))

        # Data check
        self.assertTrue(np.all(ref_row[:5] == row[:5]))
        self.assertTrue(np.all(ref_column[:5] == column[:5]))
        self.assertTrue(np.all(ref_row[6:10] == row[6:10]))
        self.assertTrue(np.all(ref_column[6:10] == column[6:10]))
        self.assertTrue(np.all(ref_row[20:] == row[20:]))
        self.assertTrue(np.all(ref_column[20:] == column[20:]))

    def test_no_correction(self):  # check behavior if no correction is needed
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(5000)
        # Check with correlation hole
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        self.assertEqual(n_fixes, 0)  # no fixes are expected
        self.assertTrue(np.all(corr == 1))  # Correlation flag check
        self.assertTrue(np.all(ref_column == column))  # Similarity check
        self.assertTrue(np.all(ref_row == row))  # Similarity check

    def test_virtual_hit_copying(self):  # check behavior for virtual hits
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(20)
        event_numbers[:12] = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])
        column[4:] = column[1:-3]
        row[4:] = row[1:-3]
        column[1:4] = 0
        row[1:4] = 0
        column[4:-1] = column[5:]
        row[4:-1] = row[5:]
        column[9:] = column[8:-1]
        row[9:] = row[8:-1]
        column[8] = 0
        row[8] = 0
        column[13:] = ref_column[11:-2]
        row[13:] = ref_row[11:-2]
        column[12] = 0
        row[12] = 0

        # Check with correlation hole
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:18] == 1))
        self.assertTrue(np.all(corr[18:] == 0))

        # Similarity check
        self.assertEqual(ref_column[0], column[0])
        self.assertEqual(ref_row[0], row[0])
        self.assertTrue(np.all(ref_column[4:9] == column[4:9]))
        self.assertTrue(np.all(ref_row[4:9] == row[4:9]))
        self.assertTrue(np.all(ref_column[11:18] == column[11:18]))
        self.assertTrue(np.all(ref_row[11:18] == row[11:18]))

        # Virtual hits check
        self.assertEqual(column[3], 0)
        self.assertEqual(row[3], 0)
        self.assertTrue(np.all(column[9:11] == 0))
        self.assertTrue(np.all(row[9:11] == 0))
        self.assertTrue(np.all(column[18:] == 0))
        self.assertTrue(np.all(row[18:] == 0))

    def test_missing_events(self):  # test behaviour if events are missing
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(20, hits_per_event=1)
        # Event offset = 3 and two consecutive events missing
        column[:3] = 0
        row[:3] = 0
        column[3:] = ref_column[:-3]
        row[3:] = ref_row[:-3]
        event_numbers = np.delete(event_numbers, [9, 10], axis=0)
        ref_column = np.delete(ref_column, [9, 10], axis=0)
        column = np.delete(column, [9, 10], axis=0)
        ref_row = np.delete(ref_row, [9, 10], axis=0)
        row = np.delete(row, [9, 10], axis=0)
        corr = np.delete(corr, [9, 10], axis=0)

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:-3] == 1))
        self.assertTrue(np.all(corr[-3:] == 0))

        # Similarity check
        self.assertTrue(np.all(ref_column[column != 0] == column[column != 0]))
        self.assertTrue(np.all(ref_row[column != 0] == row[column != 0]))
        self.assertTrue(np.all(column[6:8] == 0))
        self.assertTrue(np.all(row[6:8] == 0))

        # Event offset = 1, no events missing, but hits of one event missing
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(20, hits_per_event=3)
        column[:3] = 0
        row[:3] = 0
        column[3:] = ref_column[:-3]
        row[3:] = ref_row[:-3]
        event_numbers = np.delete(event_numbers, [9, 10], axis=0)
        ref_column = np.delete(ref_column, [9, 10], axis=0)
        column = np.delete(column, [9, 10], axis=0)
        ref_row = np.delete(ref_row, [9, 10], axis=0)
        row = np.delete(row, [9, 10], axis=0)
        corr = np.delete(corr, [9, 10], axis=0)

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:9] == 1))
        self.assertEqual(corr[9], 0)
        self.assertTrue(np.all(corr[10:14] == 1))
        self.assertTrue(np.all(corr[16:] == 0))

        # Similarity check
        self.assertTrue(np.all(ref_column[0:6] == column[0:6]))
        self.assertTrue(np.all(ref_row[0:6] == row[0:6]))
        self.assertTrue(np.all(ref_column[10:15] == column[10:15]))
        self.assertTrue(np.all(ref_row[10:15] == row[10:15]))
        self.assertTrue(np.all(column[15:] == 0))
        self.assertTrue(np.all(row[15:] == 0))

        # Event offset = 1, 1 hit events, missing hits
        event_numbers, ref_column, column, ref_row, row, corr = get_random_data(20, hits_per_event=1)

        ref_column[5] = 0
        ref_row[5] = 0

        ref_column[11:13] = 0
        ref_row[11:13] = 0

        column[:3] = 0
        row[:3] = 0
        column[3:] = ref_column[:-3]
        row[3:] = ref_row[:-3]
        corr = np.ones_like(event_numbers, dtype=np.uint8)

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:17] == 1))
        self.assertTrue(np.all(corr[17:] == 0))

        # Similarity check
        self.assertTrue(np.all(ref_column[corr == 1] == column[corr == 1]))
        self.assertTrue(np.all(ref_row[corr == 1] == row[corr == 1]))

    def test_tough_test_case(self):  # test crazy uncorrelated data
        #         raise SkipTest
        event_numbers = np.array([0, 0, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 10], dtype=np.int64)
        ref_column = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.double)
        ref_row = ref_column
        column = np.array([11, 11, 0, 2, 5, 5, 5, 3, 0, 4, 9, 10, 12, 12, 13, 14, 15, 17], dtype=np.double)
        row = column

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=0.1, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:16] == 1))
        self.assertTrue(np.all(corr[17:] == 0))

        # Similarity check
        self.assertTrue(np.all(column == np.array([2, 0, 3, 4, 0, 0, 0, 9, 10, 0, 13, 14, 15, 0, 17, 0, 0, 0])))
        self.assertTrue(np.all(column == row))

        # Small but important change of test case, event 4 is copied to 2 and their are too many hits in 4 -> correlation has to be 0
        event_numbers = np.array([0, 0, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 10], dtype=np.int64)
        ref_column = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.double)
        ref_row = ref_column
        column = np.array([11, 11, 0, 2, 5, 5, 5, 3, 3, 4, 9, 10, 12, 12, 13, 14, 15, 17], dtype=np.double)
        row = column

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[0:2] == 1))
        self.assertTrue(np.all(corr[2:4] == 0))
        self.assertTrue(np.all(corr[4:16] == 1))
        self.assertTrue(np.all(corr[17:] == 0))

        # Similarity check
        self.assertTrue(np.all(column == np.array([2, 0, 3, 3, 0, 0, 0, 9, 10, 0, 13, 14, 15, 0, 17, 0, 0, 0])))
        self.assertTrue(np.all(column == row))

if __name__ == '__main__':
    tests_data_folder = 'test_analysis//'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
