''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import matplotlib
# Force matplotlib to not use any Xwindows backend, http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg')
import unittest
import tables as tb
import numpy as np
import os


from multiprocessing import Pool
from testbeam_analysis.clusterizer import data_struct
from testbeam_analysis import analysis_utils

import testbeam_analysis.analysis as tba
from testbeam_analysis import plot_utils

tests_data_folder = r'tests/test_track_analysis/'


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


def compare_h5_files(first_file, second_file, expected_nodes=None, detailed_comparison=True, exact=True):
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
            n_expected_nodes = sum(1 for _ in enumerate(first_h5_file.root)) if expected_nodes is None else expected_nodes  # set the number of expected nodes
            n_nodes = sum(1 for _ in enumerate(second_h5_file.root))  # calculated the number of nodes
            if n_nodes != n_expected_nodes:
                checks_passed = False
                error_msg += 'The number of nodes in the file is wrong.\n'
            for node in second_h5_file.root:  # loop over all nodes and compare each node, do not abort if one node is wrong
                node_name = node.name
                try:
                    expected_data = first_h5_file.get_node(first_h5_file.root, node_name)[:]
                    data = second_h5_file.get_node(second_h5_file.root, node_name)[:]
                    if (exact or expected_data.dtype.names is not None):  # exact comparison if exact is set and on recarray data (np.allclose does not work on recarray)
                        if not np.all(expected_data == data):  # compare the arrays for each element
                            checks_passed = False
                            error_msg += node_name
                            if detailed_comparison:
                                error_msg += get_array_differences(expected_data, data)
                            error_msg += '\n'
                    else:
                        if not np.allclose(expected_data, data):
                            np.allclose(expected_data, data)
                            checks_passed = False
                            error_msg += node_name
                            if detailed_comparison:
                                error_msg += get_array_differences(expected_data, data)
                            error_msg += '\n'
                except tb.NoSuchNodeError:
                    checks_passed = False
                    error_msg += 'Unknown node ' + node_name + '\n'
    return checks_passed, error_msg


class TestTrackAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_folder = tests_data_folder
        cls.pixel_size = (250, 50)  # in um
        cls.z_positions = [0., 1.95, 10.88, 12.83]

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(cls.output_folder + 'TrackCandidates.h5')
        os.remove(cls.output_folder + 'Tracks.h5')

    def test_track_finding(self):
        tba.find_tracks(tracklets_file=tests_data_folder + 'Tracklets_small.h5',
                        alignment_file=tests_data_folder + r'Alignment_result.h5',
                        track_candidates_file=self.output_folder + 'TrackCandidates.h5',
                        pixel_size=self.pixel_size)
        compare_h5_files(tests_data_folder + 'TrackCandidates_result.h5', self.output_folder + 'TrackCandidates.h5')

    def test_track_fitting(self):
        # Fit the track candidates and create new track table
        tba.fit_tracks(track_candidates_file=tests_data_folder + 'TrackCandidates_result.h5',
                       tracks_file=self.output_folder + 'Tracks.h5',
                       output_pdf=self.output_folder + 'Tracks.pdf',
                       z_positions=self.z_positions,
                       fit_duts=None,
                       include_duts=[-3, -2, -1, 1, 2, 3],
                       ignore_duts=None,
                       max_tracks=1,
                       track_quality=1,
                       pixel_size=self.pixel_size,
                       use_correlated=False)
        compare_h5_files(tests_data_folder + 'Tracks_result.h5', self.output_folder + 'Tracks.h5')

if __name__ == '__main__':
    tests_data_folder = r'test_track_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
