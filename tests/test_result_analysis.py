''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import tables as tb
import numpy as np
import os


from multiprocessing import Pool
from testbeam_analysis.clusterizer import data_struct
from testbeam_analysis import analysis_utils

import testbeam_analysis.analysis as tba
from testbeam_analysis import plot_utils

tests_data_folder = r'tests/test_result_analysis/'


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
                return_str += 'Column ' + column_name + \
                    ' has different data type. '
            try:
                # check if the data of the column is equal
                if not (first_column == second_column).all():
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
            n_expected_nodes = sum(1 for _ in enumerate(
                first_h5_file.root)) if expected_nodes is None else expected_nodes  # set the number of expected nodes
            # calculated the number of nodes
            n_nodes = sum(1 for _ in enumerate(second_h5_file.root))
            if n_nodes != n_expected_nodes:
                checks_passed = False
                error_msg += 'The number of nodes in the file is wrong.\n'
            # loop over all nodes and compare each node, do not abort if one
            # node is wrong
            for node in second_h5_file.root:
                node_name = node.name
                try:
                    expected_data = first_h5_file.get_node(
                        first_h5_file.root, node_name)[:]
                    data = second_h5_file.get_node(
                        second_h5_file.root, node_name)[:]
                    # exact comparison if exact is set and on recarray data
                    # (np.allclose does not work on recarray)
                    if (exact or expected_data.dtype.names is not None):
                        # compare the arrays for each element
                        if not np.all(expected_data == data):
                            checks_passed = False
                            error_msg += node_name
                            if detailed_comparison:
                                error_msg += get_array_differences(
                                    expected_data, data)
                            error_msg += '\n'
                    else:
                        if not np.allclose(expected_data, data):
                            np.allclose(expected_data, data)
                            checks_passed = False
                            error_msg += node_name
                            if detailed_comparison:
                                error_msg += get_array_differences(
                                    expected_data, data)
                            error_msg += '\n'
                except tb.NoSuchNodeError:
                    checks_passed = False
                    error_msg += 'Unknown node ' + node_name + '\n'
    return checks_passed, error_msg


class TestResultAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            from xvfbwrapper import Xvfb
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.output_folder = tests_data_folder
        cls.pixel_size = (250, 50)  # in um
        cls.z_positions = [0., 1.95, 10.88, 12.83]  # in cm

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass
        os.remove(cls.output_folder + 'Efficiency.pdf')
        os.remove(cls.output_folder + 'Residuals.pdf')

    def test_residuals_calculation(self):
        residuals = tba.calculate_residuals(tracks_file=tests_data_folder + 'Tracks_result.h5',
                                            output_pdf=self.output_folder + 'Residuals.pdf',
                                            z_positions=self.z_positions,
                                            pixel_size=self.pixel_size,
                                            use_duts=None,
                                            max_chi2=3e3)

        # Only test row residuals, columns are to large (250 um) for meaningfull gaussian residuals distribution
        self.assertAlmostEqual(residuals[1], 31.012493732525773, msg='DUT 0 row residuals do not match')
        self.assertAlmostEqual(residuals[3], 21.480965949719472, msg='DUT 1 row residuals do not match')
        self.assertAlmostEqual(residuals[5], 31.372070715636198, msg='DUT 2 row residuals do not match')
        self.assertAlmostEqual(residuals[7], 44.449772465251371, msg='DUT 3 row residuals do not match')

    def test_efficiency_calculation(self):
        efficiencies = tba.calculate_efficiency(tracks_file=self.output_folder + 'Tracks_result.h5',
                                                output_pdf=self.output_folder + 'Efficiency.pdf',
                                                z_positions=self.z_positions,
                                                dim_x=80,
                                                dim_y=336,
                                                minimum_track_density=2,
                                                pixel_size=self.pixel_size,
                                                use_duts=None,
                                                cut_distance=500,
                                                max_distance=500,
                                                col_range=(5, 70),
                                                row_range=(20, 320))
        self.assertAlmostEqual(efficiencies[0], 100, msg='DUT 0 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[1], 98.113207547169807, msg='DUT 1 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[2], 97.484276729559738, msg='DUT 2 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[3], 92.666666666666671, msg='DUT 3 efficiencies do not match')

if __name__ == '__main__':
    tests_data_folder = r'test_result_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResultAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
