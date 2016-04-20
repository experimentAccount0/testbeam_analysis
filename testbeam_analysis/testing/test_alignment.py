''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

from testbeam_analysis import dut_alignment
from testbeam_analysis.tools import test_tools

tests_data_folder = r'tests/test_dut_alignment/'


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            try:
                from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
                cls.vdisplay = Xvfb()
                cls.vdisplay.start()
            except (ImportError, EnvironmentError):
                pass
        cls.data_files = [os.path.join(tests_data_folder + 'Cluster_DUT0_cluster.h5'),
                          os.path.join(tests_data_folder + 'Cluster_DUT1_cluster.h5'),
                          os.path.join(tests_data_folder + 'Cluster_DUT2_cluster.h5'),
                          os.path.join(tests_data_folder + 'Cluster_DUT3_cluster.h5')
                          ]
        cls.output_folder = tests_data_folder
        cls.n_pixels = [(80, 336)] * 4
        cls.pixel_size = [(250, 50)] * 4  # in um
        cls.z_positions = [0., 19500, 108800, 128300]

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(os.path.join(cls.output_folder + 'Correlation.h5'))
        os.remove(os.path.join(cls.output_folder + 'Correlation.pdf'))
        os.remove(os.path.join(cls.output_folder + 'Tracklets.h5'))
        os.remove(os.path.join(cls.output_folder + 'Tracklets_2.h5'))
        os.remove(cls.output_folder + 'Alignment.h5')
        os.remove(cls.output_folder + 'Alignment.pdf')

    def test_cluster_correlation(self):  # check the hit correlation function
        dut_alignment.correlate_cluster(input_cluster_files=self.data_files,
                                        output_correlation_file=os.path.join(self.output_folder + 'Correlation.h5'),
                                        n_pixels=self.n_pixels,
                                        pixel_size=self.pixel_size
                                        )
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Correlation_result.h5'), os.path.join(self.output_folder + 'Correlation.h5'), exact=True)
        self.assertTrue(data_equal, msg=error_msg)
# FIXME: chunking does not work
#         # Retest with tiny chunk size to force chunked correlation
#         dut_alignment.correlate_cluster(input_cluster_files=self.data_files,
#                                         output_correlation_file=os.path.join(self.output_folder + 'Correlation_2.h5'),
#                                         n_pixels=self.n_pixels,
#                                         pixel_size=self.pixel_size,
#                                         chunk_size=293
#                                         )
#         data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Correlation_result.h5'), os.path.join(self.output_folder + 'Correlation_2.h5'), exact=True)
#         self.assertTrue(data_equal, msg=error_msg)

    def test_prealignment(self):  # Check the hit alignment function
        dut_alignment.prealignment(input_correlation_file=os.path.join(tests_data_folder + 'Correlation_result.h5'),
                                   output_alignment_file=os.path.join(self.output_folder + 'Alignment.h5'),
                                   z_positions=self.z_positions,
                                   pixel_size=self.pixel_size,
                                   non_interactive=True,
                                   iterations=5)  # Due too to little test data the alignment result is only stable for more iterations
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Alignment_result.h5'),
                                                            os.path.join(self.output_folder + 'Alignment.h5'),
                                                            exact=False,
                                                            rtol=0.05,  # 5 % error allowed
                                                            atol=5)  # 5 um absolute tolerance allowed
        self.assertTrue(data_equal, msg=error_msg)

    def test_cluster_merging(self):
        cluster_files = [os.path.join(tests_data_folder + 'Cluster_DUT%d_cluster.h5') % i for i in range(4)]
        dut_alignment.merge_cluster_data(cluster_files,
                                         output_merged_file=os.path.join(self.output_folder + 'Tracklets.h5'),
                                         pixel_size=self.pixel_size)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Tracklets_result.h5'), os.path.join(self.output_folder + 'Tracklets.h5'))
        self.assertTrue(data_equal, msg=error_msg)
# FIXME: chunking does not work
#         # Retest with tiny chunk size to force chunked merging
#         dut_alignment.merge_cluster_data(cluster_files,
#                                          output_merged_file=os.path.join(self.output_folder + 'Tracklets_2.h5'),
#                                          pixel_size=self.pixel_size,
#                                          chunk_size=293)
# 
#         data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Tracklets_result.h5', self.output_folder + 'Tracklets_2.h5')
#         self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    tests_data_folder = r'test_dut_alignment/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
