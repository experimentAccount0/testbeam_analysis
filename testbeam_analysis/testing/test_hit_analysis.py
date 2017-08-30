''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

from testbeam_analysis import hit_analysis
from testbeam_analysis.tools import test_tools

# Get package path
testing_path = os.path.dirname(__file__)  # Get the absoulte path of the online_monitor installation

test_tools.install_quilt_data(package='DavidLP/tba_fixtures',
                              hash='ecc02538576cb905044222270674ef57e21e44c67d6bab36d4c0eb6ce08f4e66')
from quilt.data.DavidLP import tba_fixtures


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.getenv('TRAVIS', False):
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.noisy_data_file = test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'TestBeamData_Mimosa26_DUT0_small')
        cls.data_file = test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'TestBeamData_FEI4_DUT0_small')
        cls.output_folder = ''
        cls.pixel_size = ((250, 50), (250, 50), (250, 50), (250, 50))  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_FEI4_DUT0_small_clustered.h5'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_FEI4_DUT0_small_clustered_cluster_size.pdf'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixel_mask.h5'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixel_mask_masked_pixels.pdf'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_disabled_pixel_mask.h5'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_disabled_pixel_mask_masked_pixels.pdf'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_clustered.h5'))
        os.remove(os.path.join(cls.output_folder, 'TestBeamData_Mimosa26_DUT0_small_clustered_cluster_size.pdf'))

    def test_noisy_pixel_masking(self):
        # Test 1:
        output_mask_file = hit_analysis.generate_pixel_mask(input_hits_file=self.noisy_data_file,
                                                            output_mask_file=os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixel_mask.h5'),
                                                            pixel_mask_name="NoisyPixelMask",
                                                            threshold=10.0, n_pixel=(1152, 576),
                                                            pixel_size=(18.4, 18.4), plot=True)
        output_cluster_file = hit_analysis.cluster_hits(input_hits_file=self.noisy_data_file,
                                                        create_cluster_hits_table=False,
                                                        input_noisy_pixel_mask_file=output_mask_file,
                                                        min_hit_charge=1, max_hit_charge=1,
                                                        column_cluster_distance=2, row_cluster_distance=2,
                                                        frame_cluster_distance=1)
        data_equal, error_msg = test_tools.compare_h5_files(test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'Mimosa26_noisy_pixels_cluster_result'), output_cluster_file, exact=False)
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: smaller chunks
        output_mask_file = hit_analysis.generate_pixel_mask(input_hits_file=self.noisy_data_file,
                                                            output_mask_file=os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_noisy_pixel_mask.h5'),
                                                            pixel_mask_name="NoisyPixelMask",
                                                            threshold=10.0, n_pixel=(1152, 576),
                                                            pixel_size=(18.4, 18.4), plot=True)
        output_cluster_file = hit_analysis.cluster_hits(input_hits_file=self.noisy_data_file,
                                                        create_cluster_hits_table=False,
                                                        input_noisy_pixel_mask_file=output_mask_file,
                                                        min_hit_charge=1, max_hit_charge=1, column_cluster_distance=2,
                                                        row_cluster_distance=2, frame_cluster_distance=1, chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'Mimosa26_noisy_pixels_cluster_result'), output_cluster_file, exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    def test_noisy_pixel_remover(self):
        # Test 1:
        output_mask_file = hit_analysis.generate_pixel_mask(input_hits_file=self.noisy_data_file,
                                                            output_mask_file=os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_disabled_pixel_mask.h5'),
                                                            pixel_mask_name="DisabledPixelMask",
                                                            threshold=10.0, n_pixel=(1152, 576),
                                                            pixel_size=(18.4, 18.4), plot=True)
        output_cluster_file = hit_analysis.cluster_hits(input_hits_file=self.noisy_data_file,
                                                        output_cluster_file=os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_clustered.h5'),
                                                        create_cluster_hits_table=False,
                                                        input_disabled_pixel_mask_file=output_mask_file,
                                                        min_hit_charge=1, max_hit_charge=1, column_cluster_distance=2,
                                                        row_cluster_distance=2, frame_cluster_distance=1)
        data_equal, error_msg = test_tools.compare_h5_files(test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'Mimosa26_disabled_pixels_cluster_result'), output_cluster_file, exact=False)
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: smaller chunks
        output_mask_file = hit_analysis.generate_pixel_mask(input_hits_file=self.noisy_data_file,
                                                            output_mask_file=os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_disabled_pixel_mask.h5'),
                                                            pixel_mask_name="DisabledPixelMask",
                                                            threshold=10.0, n_pixel=(1152, 576), pixel_size=(18.4, 18.4), plot=True)
        output_cluster_file = hit_analysis.cluster_hits(input_hits_file=self.noisy_data_file,
                                                        output_cluster_file=os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_small_clustered.h5'),
                                                        create_cluster_hits_table=False,
                                                        input_disabled_pixel_mask_file=output_mask_file,
                                                        min_hit_charge=1, max_hit_charge=1,
                                                        column_cluster_distance=2, row_cluster_distance=2, frame_cluster_distance=1,
                                                        chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'Mimosa26_disabled_pixels_cluster_result'), output_cluster_file, exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    def test_hit_clustering(self):
        # Test 1:
        output_cluster_file = hit_analysis.cluster_hits(input_hits_file=self.data_file, min_hit_charge=0, max_hit_charge=13,
                                                        output_cluster_file=os.path.join(self.output_folder, 'TestBeamData_FEI4_DUT0_small_clustered.h5'),
                                                        column_cluster_distance=1, row_cluster_distance=2, frame_cluster_distance=2)
        data_equal, error_msg = test_tools.compare_h5_files(test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'FEI4_cluster_result'), output_cluster_file, exact=False)
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: smaller chunks
        output_cluster_file = hit_analysis.cluster_hits(input_hits_file=self.data_file, min_hit_charge=0, max_hit_charge=13,
                                                        output_cluster_file=os.path.join(self.output_folder, 'TestBeamData_FEI4_DUT0_small_clustered.h5'),
                                                        column_cluster_distance=1, row_cluster_distance=2, frame_cluster_distance=2,
                                                        chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(test_tools.get_quilt_data(tba_fixtures.hit_analysis, 'FEI4_cluster_result'), output_cluster_file, exact=False)
        self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
