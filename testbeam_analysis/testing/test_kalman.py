''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

import numpy as np

from testbeam_analysis import track_analysis
from testbeam_analysis.tools import test_tools

# Get package path
testing_path = os.path.dirname(__file__)  # Get the absoulte path of the online_monitor installation

# Set the converter script path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/fixtures/track_analysis/'))


class TestTrackAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.getenv('TRAVIS', False):
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        # cls.output_folder = tests_data_folder
        # cls.pixel_size = (250, 50)  # in um

    @classmethod
    def tearDownClass(cls):  # Remove created files
        pass
        # os.remove(os.path.join(cls.output_folder, 'Tracks_merged.pdf'))

    def test_kalman(self):
        # pixel size of sensor
        pixel_size = np.array([(18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (250., 50.)])
        pixel_resolution = pixel_size / np.sqrt(12)
        material_budget = np.array([100., 100., 100., 100., 100., 100., 250.]) / np.array([125390., 125390., 125390., 125390., 125390., 125390., 93700.])
        kwargs = {'track_hits': np.array([[[-1229.22372954, 2828.19616302, 0., pixel_resolution[0][0], pixel_resolution[0][1]],
                                           [np.nan, np.nan, np.nan, np.nan, np.nan],  # [-1254.51224282, 2827.4291421, 29900.],
                                           [-1285.6117892, 2822.34536687, 60300., pixel_resolution[2][0], pixel_resolution[2][1]],
                                           [-1311.31083616, 2823.56121414, 82100., pixel_resolution[3][0], pixel_resolution[3][1]],
                                           [-1335.8529645, 2828.43359043, 118700., pixel_resolution[4][0], pixel_resolution[4][1]],
                                           [-1357.81872222, 2840.86947964, 160700., pixel_resolution[5][0], pixel_resolution[5][1]],
                                           [-1396.35698339, 2843.76799577, 197800., pixel_resolution[6][0], pixel_resolution[6][1]]]]),
                  'dut_fit_selection': 61,
                  'z_positions': [[0., 29900, 60300, 82100, 118700, 160700, 197800]],
                  'alignment': [None],
                  'pixel_size': pixel_size,
                  'n_pixels': ((576, 1152), (576, 1152), (576, 1152), (576, 1152), (576, 1152), (576, 1152), (80, 336)),
                  'beam_energy': 2500.,
                  'material_budget': material_budget,
                  'add_scattering_plane': None}

        # expected result array: (state estimates, chi, x error, y errors)
        result = np.array([[[-1.23045812e+03, 2.82684464e+03, 9.54189393e-04, 5.78723069e-05],
                            [-1.25900270e+03, 2.82511339e+03, 9.54667995e-04, 5.79013344e-05],
                            [-1.28705254e+03, 2.82443254e+03, 9.22692240e-04, 2.23966275e-05],
                            [-1.30575083e+03, 2.82550588e+03, 8.57719412e-04, -4.92360235e-05],
                            [-1.33339390e+03, 2.83014572e+03, 7.55275169e-04, -1.26771525e-04],
                            [-1.36192826e+03, 2.83782855e+03, 6.79389545e-04, -1.82924543e-04],
                            [-1.38713361e+03, 2.84461505e+03, 6.79389545e-04, -1.82924543e-04]],
                           [74],
                           [3.62429044, 3.2884327, 3.21655702, 3.1539946, 3.23671172, 4.66501707, 8.62909928],
                           [3.62429044, 3.2884327, 3.21655702, 3.1539946, 3.23671172, 4.66501707, 8.62909928]])

        for i in range(4):  # test each return (state estimates, chi, x error, y errors) seperatly
            test = test_tools._call_function_with_args(function=track_analysis._fit_tracks_kalman_loop,
                                                       **kwargs)[0][i]
            data_equal = test_tools.array_close(test, result[i])
            self.assertTrue(data_equal)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
