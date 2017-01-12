''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

from testbeam_analysis import hit_analysis
from testbeam_analysis.tools import test_tools

# Get package path
testing_path = os.path.dirname(__file__)  # Get the absoulte path of the online_monitor installation

# Set the converter script path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/fixtures/hit_analysis/'))


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.getenv('TRAVIS', False):
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.noisy_data_file = os.path.join(tests_data_folder, 'TestBeamData_Mimosa26_DUT0_small.h5')
        cls.data_files = [os.path.join(tests_data_folder, 'TestBeamData_FEI4_DUT0_small.h5'),
                          os.path.join(tests_data_folder, 'TestBeamData_FEI4_DUT1_small.h5'),
                          os.path.join(tests_data_folder, 'TestBeamData_FEI4_DUT2_small.h5'),
                          os.path.join(tests_data_folder, 'TestBeamData_FEI4_DUT3_small.h5')
                          ]
        cls.output_folder = tests_data_folder
        cls.pixel_size = ((250, 50), (250, 50), (250, 50), (250, 50))  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_FEI4_DUT0_small_cluster.h5'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixels.h5'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixels.pdf'))

    def test_noisy_pixel_remover(self):
        # Test 1:
        hit_analysis.remove_noisy_pixels(self.noisy_data_file, threshold=10.0, n_pixel=(1152, 576), pixel_size=(18.4, 18.4))
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder, 'HotPixel_result.h5'), os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixels.h5'))
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: smaller chunks
        hit_analysis.remove_noisy_pixels(self.noisy_data_file, threshold=10.0, n_pixel=(1152, 576), pixel_size=(18.4, 18.4), chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder, 'HotPixel_result.h5'), os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixels.h5'))
        self.assertTrue(data_equal, msg=error_msg)

    def test_hit_clustering(self):
        # Test 1:
        hit_analysis.cluster_hits(self.data_files[0], max_x_distance=1, max_y_distance=2)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder, 'Cluster_result.h5'), os.path.join(self.output_folder, 'TestBeamData_FEI4_DUT0_small_cluster.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: smaller chunks
        hit_analysis.cluster_hits(self.data_files[0], max_x_distance=1, max_y_distance=2, chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder, 'Cluster_result.h5'), os.path.join(self.output_folder, 'TestBeamData_FEI4_DUT0_small_cluster.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
