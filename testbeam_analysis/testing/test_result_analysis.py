''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

from testbeam_analysis import result_analysis

# Get package path
testing_path = os.path.dirname(__file__)  # Get the absoulte path of the online_monitor installation

# Set the converter script path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/fixtures/result_analysis/'))


class TestResultAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.getenv('TRAVIS', False):
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.output_folder = tests_data_folder
        cls.pixel_size = [250, 50] * 4  # in um
        cls.n_pixels = [80, 336] * 4
        cls.z_positions = [0., 19500, 108800, 128300]  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass
#         os.remove(os.path.join(cls.output_folder, 'Efficiency.pdf'))
#         os.remove(os.path.join(cls.output_folder, 'Residuals.pdf'))

    @unittest.SkipTest
    def test_residuals_calculation(self):
        residuals = result_analysis.calculate_residuals(input_tracks_file=os.path.join(tests_data_folder, 'Tracks_result.h5'),
                                                        input_alignment_file=os.path.join(tests_data_folder, r'Alignment_result.h5'),
                                                        output_residuals_file=os.path.join(self.output_folder, 'Residuals.h5'),
                                                        n_pixels=self.n_pixels,
                                                        pixel_size=self.pixel_size,
                                                        max_chi2=10000)
        # Only test row residuals, columns are too large (250 um) for meaningfull gaussian residuals distribution
        self.assertAlmostEqual(residuals[1], 22.9135, msg='DUT 0 row residuals do not match', places=3)
        self.assertAlmostEqual(residuals[3], 18.7317, msg='DUT 1 row residuals do not match', places=3)
        self.assertAlmostEqual(residuals[5], 22.8645, msg='DUT 2 row residuals do not match', places=3)
        self.assertAlmostEqual(residuals[7], 27.2816, msg='DUT 3 row residuals do not match', places=3)

    @unittest.SkipTest
    def test_efficiency_calculation(self):
        efficiencies = result_analysis.calculate_efficiency(tracks_file=os.path.join(self.output_folder, 'Tracks_result.h5'),
                                                            output_pdf=os.path.join(self.output_folder, r'Efficiency.pdf'),
                                                            z_positions=self.z_positions,
                                                            bin_size=(250, 50),
                                                            minimum_track_density=2,
                                                            use_duts=None,
                                                            cut_distance=500,
                                                            max_distance=500,
                                                            col_range=(1250, 17500),
                                                            row_range=(1000, 16000))

        self.assertAlmostEqual(efficiencies[0], 100.000, msg='DUT 0 efficiencies do not match', places=3)
        self.assertAlmostEqual(efficiencies[1], 98.7013, msg='DUT 1 efficiencies do not match', places=3)
        self.assertAlmostEqual(efficiencies[2], 97.4684, msg='DUT 2 efficiencies do not match', places=3)
        self.assertAlmostEqual(efficiencies[3], 100.000, msg='DUT 3 efficiencies do not match', places=3)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResultAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
