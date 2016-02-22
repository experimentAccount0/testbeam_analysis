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
        cls.data_files = [os.path.join(tests_data_folder + 'TestBeamData_FEI4_DUT0_small.h5'),
                          os.path.join(tests_data_folder + 'TestBeamData_FEI4_DUT1_small.h5'),
                          os.path.join(tests_data_folder + 'TestBeamData_FEI4_DUT2_small.h5'),
                          os.path.join(tests_data_folder + 'TestBeamData_FEI4_DUT3_small.h5')
                          ]
        cls.output_folder = tests_data_folder
        cls.pixel_size = ((250, 50), (250, 50), (250, 50), (250, 50))  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(os.path.join(cls.output_folder + 'Correlation.h5'))
#         os.remove(cls.output_folder + 'Alignment.h5')
#         os.remove(cls.output_folder + 'Alignment.pdf')
        os.remove(os.path.join(cls.output_folder + 'Tracklets.h5'))
        os.remove(os.path.join(cls.output_folder + 'Tracklets_2.h5'))

    def test_hit_correlation(self):  # check the hit correlation function
        dut_alignment.correlate_hits(input_hits_files=self.data_files,
                                     output_correlation_file=os.path.join(self.output_folder + 'Correlation.h5'),
                                     fraction=1,
                                     event_range=0)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Correlation_result.h5'), os.path.join(self.output_folder + 'Correlation.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    @unittest.SkipTest  # Unclear how to check interactive alignment automatically
    def test_coarse_alignment(self):  # Check the hit alignment function
        dut_alignment.coarse_alignment(input_correlation_file=os.path.join(tests_data_folder + 'Correlation_result.h5'),
                                       output_alignment_file=os.path.join(self.output_folder + 'Alignment.h5'),
                                       output_pdf=os.path.join(self.output_folder + 'Alignment.pdf'),
                                       pixel_size=self.pixel_size)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Alignment_result.h5'), os.path.join(self.output_folder + 'Alignment.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    def test_cluster_merging(self):
        cluster_files = [os.path.join(tests_data_folder + 'Cluster_DUT%d_cluster.h5') % i for i in range(4)]
        dut_alignment.merge_cluster_data(cluster_files,
                                         input_alignment_file=os.path.join(tests_data_folder + 'Alignment_result.h5'),
                                         output_tracklets_file=os.path.join(self.output_folder + 'Tracklets.h5'),
                                         pixel_size=self.pixel_size)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(tests_data_folder + 'Tracklets_result.h5'), os.path.join(self.output_folder + 'Tracklets.h5'))
        self.assertTrue(data_equal, msg=error_msg)
        # Retest with tiny chunk size to force chunked merging
        dut_alignment.merge_cluster_data(cluster_files,
                                         input_alignment_file=os.path.join(tests_data_folder + 'Alignment_result.h5'),
                                         output_tracklets_file=os.path.join(self.output_folder + 'Tracklets_2.h5'),
                                         pixel_size=self.pixel_size,
                                         chunk_size=293)

        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Tracklets_result.h5', self.output_folder + 'Tracklets_2.h5')
        self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    tests_data_folder = r'test_dut_alignment/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
