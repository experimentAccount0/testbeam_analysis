''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import tables as tb
import numpy as np
import os

from testbeam_analysis import hit_analysis

tests_data_folder = r'tests/test_hit_analysis/'


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
                        if not np.array_equal(expected_data, data):  # compare the arrays for each element
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


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.noisy_data_file = tests_data_folder + 'TestBeamData_Mimosa26_DUT0_small.h5'
        cls.data_files = [tests_data_folder + 'TestBeamData_FEI4_DUT0_small.h5',
                          tests_data_folder + 'TestBeamData_FEI4_DUT1_small.h5',
                          tests_data_folder + 'TestBeamData_FEI4_DUT2_small.h5',
                          tests_data_folder + 'TestBeamData_FEI4_DUT3_small.h5'
                          ]
        cls.output_folder = tests_data_folder
        cls.pixel_size = ((250, 50), (250, 50), (250, 50), (250, 50))  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(cls.output_folder + 'TestBeamData_FEI4_DUT0_small_cluster.h5')
        os.remove(cls.output_folder + 'TestBeamData_Mimosa26_DUT0_small_hot_pixel.h5')
        os.remove(cls.output_folder + 'TestBeamData_Mimosa26_DUT0_small_hot_pixel.pdf')

    def test_noisy_pixel_remover(self):
        hit_analysis.remove_noisy_pixels(self.noisy_data_file)
        data_equal, error_msg = compare_h5_files(tests_data_folder + 'HotPixel_result.h5', self.output_folder + 'TestBeamData_Mimosa26_DUT0_small_hot_pixel.h5')
        self.assertTrue(data_equal, msg=error_msg)

    def test_hit_clustering(self):
        hit_analysis.cluster_hits(self.data_files[0], n_cols=80, n_rows=336, n_frames=16, n_charges=16, max_x_distance=1, max_y_distance=2)
        data_equal, error_msg = compare_h5_files(tests_data_folder + 'Cluster_result.h5', self.output_folder + 'TestBeamData_FEI4_DUT0_small_cluster.h5', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    tests_data_folder = r'test_hit_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
