''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import os

from testbeam_analysis import hit_analysis
from testbeam_analysis.tools import test_tools

tests_data_folder = r'tests/test_hit_analysis/'


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
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'HotPixel_result.h5', self.output_folder + 'TestBeamData_Mimosa26_DUT0_small_hot_pixel.h5')
        self.assertTrue(data_equal, msg=error_msg)

    def test_hit_clustering(self):
        hit_analysis.cluster_hits(self.data_files[0], n_cols=80, n_rows=336, n_frames=16, n_charges=16, max_x_distance=1, max_y_distance=2)
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Cluster_result.h5', self.output_folder + 'TestBeamData_FEI4_DUT0_small_cluster.h5', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

#     def test_hit_clustering_2(self):
#         hit_analysis.cluster_hits_new(self.data_files[0], max_x_distance=1, max_y_distance=2)
#         data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Cluster_result.h5', self.output_folder + 'TestBeamData_FEI4_DUT0_small_cluster.h5', exact=False)
#         self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    tests_data_folder = r'test_hit_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
