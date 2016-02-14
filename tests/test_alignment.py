''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import os

from testbeam_analysis import dut_alignment
from testbeam_analysis.tools import test_tools

tests_data_folder = r'tests/test_dut_alignment/'


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.data_files = [tests_data_folder + 'TestBeamData_FEI4_DUT0_small.h5',
                          tests_data_folder + 'TestBeamData_FEI4_DUT1_small.h5',
                          tests_data_folder + 'TestBeamData_FEI4_DUT2_small.h5',
                          tests_data_folder + 'TestBeamData_FEI4_DUT3_small.h5'
                          ]
        cls.output_folder = tests_data_folder
        cls.pixel_size = ((250, 50), (250, 50), (250, 50), (250, 50))  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(cls.output_folder + 'Correlation.h5')
#         os.remove(cls.output_folder + 'Alignment.h5')
#         os.remove(cls.output_folder + 'Alignment.pdf')
        os.remove(cls.output_folder + 'Tracklets.h5')
        os.remove(cls.output_folder + 'Tracklets_2.h5')

    def test_hit_correlation(self):  # check the hit correlation function
        dut_alignment.correlate_hits(self.data_files,
                                     alignment_file=self.output_folder + 'Correlation.h5',
                                     fraction=1,
                                     event_range=0)
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Correlation_result.h5', self.output_folder + 'Correlation.h5', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    @unittest.SkipTest  # Unclear how to check interactive alignment automatically
    def test_hit_alignment(self):  # Check the hit alignment function
        dut_alignment.align_hits(correlation_file=tests_data_folder + 'Correlation_result.h5',
                                 alignment_file=self.output_folder + 'Alignment.h5',
                                 output_pdf=self.output_folder + 'Alignment.pdf',
                                 pixel_size=self.pixel_size)
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Alignment_result.h5', self.output_folder + 'Alignment.h5', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    def test_cluster_merging(self):
        cluster_files = [tests_data_folder + 'Cluster_DUT%d_cluster.h5' % i for i in range(4)]
        dut_alignment.merge_cluster_data(cluster_files,
                                         alignment_file=tests_data_folder + 'Alignment_result.h5',
                                         tracklets_file=self.output_folder + 'Tracklets.h5',
                                         pixel_size=self.pixel_size)
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Tracklets_result.h5', self.output_folder + 'Tracklets.h5')
        self.assertTrue(data_equal, msg=error_msg)
        # Retest with tiny chunk size to force chunked merging
        dut_alignment.merge_cluster_data(cluster_files,
                                         alignment_file=tests_data_folder + 'Alignment_result.h5',
                                         tracklets_file=self.output_folder + 'Tracklets_2.h5',
                                         pixel_size=self.pixel_size,
                                         chunk_size=293)

        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Tracklets_result.h5', self.output_folder + 'Tracklets_2.h5')
        self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    tests_data_folder = r'C:\\Users\DavidLP\\git\\testbeam_analysis\\tests\\test_dut_alignment\\'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
