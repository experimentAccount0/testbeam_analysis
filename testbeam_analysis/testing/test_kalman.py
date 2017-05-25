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
        kwargs = {'track_hits': np.array([[[-1229.22372954, 2828.19616302, 0.],
                                           [np.nan, np.nan, np.nan],  # [-1254.51224282, 2827.4291421, 29900.],
                                           [-1285.6117892, 2822.34536687, 60300.],
                                           [-1311.31083616, 2823.56121414, 82100.],
                                           [-1335.8529645, 2828.43359043, 118700.],
                                           [-1357.81872222, 2840.86947964, 160700.],
                                           [-1396.35698339, 2843.76799577, 197800.]]]),
                  'dut_fit_selection': 61,
                  'z_positions': [[0., 29900, 60300, 82100, 118700, 160700, 197800]],
                  'pixel_size': ((18.5, 18.5), (18.5, 18.5), (18.5, 18.5),
                                 (18.5, 18.5), (18.5, 18.5), (250, 50), (250, 50)),
                  'beam_energy': 2500.,
                  'total_thickness': [[100., 100., 100., 100., 100., 100., 250.]],
                  'radiation_length': [[125390., 125390., 125390., 125390., 125390., 125390., 93700.]]}

        # expected result array: (state estimates, chi, x error, y errors)
        result = np.array([[[-1.22922375e+03, 2.82819616e+03, 9.53048967e-04, 5.96680029e-05],
                            [-1.25771995e+03, 2.82641209e+03, 9.53050162e-04, 5.96680777e-05],
                            [-1.28669270e+03, 2.82459818e+03, 9.38989287e-04, 2.97608805e-05],
                            [-1.30661393e+03, 2.82511648e+03, 9.13818199e-04, -2.37752913e-05],
                            [-1.33842616e+03, 2.82786418e+03, 8.69186632e-04, -7.50739214e-05],
                            [-1.37485963e+03, 2.83196293e+03, 8.67463483e-04, -9.75893170e-05],
                            [-1.40704252e+03, 2.83558350e+03, 8.67463483e-04, -9.75893170e-05]],
                          [405.],
                          [3.77629595, 3.00521807, 3.22523435, 3.20152958, 4.56468792, 9.49981871, 15.38928532],
                          [3.77629595, 3.00434937, 3.22194104, 3.1832628,  4.10202498, 7.98372842, 13.19036558]])
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
