''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import os

from testbeam_analysis import result_analysis


tests_data_folder = r'tests/test_result_analysis/'


class TestResultAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            from xvfbwrapper import Xvfb
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.output_folder = tests_data_folder
        cls.pixel_size = (250, 50)  # in um
        cls.z_positions = [0., 19500, 108800, 128300]  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(cls.output_folder + 'Efficiency.pdf')
        os.remove(cls.output_folder + 'Residuals.pdf')

    def test_residuals_calculation(self):
        residuals = result_analysis.calculate_residuals(tracks_file=tests_data_folder + 'Tracks_result.h5',
                                                        output_pdf=self.output_folder + 'Residuals.pdf',
                                                        z_positions=self.z_positions,
                                                        use_duts=None,
                                                        max_chi2=10000)

        # Only test row residuals, columns are too large (250 um) for meaningfull gaussian residuals distribution
        self.assertAlmostEqual(residuals[1], 22.913569705886552, msg='DUT 0 row residuals do not match')
        self.assertAlmostEqual(residuals[3], 18.73170503500376, msg='DUT 1 row residuals do not match')
        self.assertAlmostEqual(residuals[5], 22.864586411335129, msg='DUT 2 row residuals do not match')
        self.assertAlmostEqual(residuals[7], 27.28165821984965, msg='DUT 3 row residuals do not match')

    def test_efficiency_calculation(self):
        efficiencies = result_analysis.calculate_efficiency(tracks_file=self.output_folder + 'Tracks_result.h5',
                                                            output_pdf=self.output_folder + r'/Efficiency.pdf',
                                                            z_positions=self.z_positions,
                                                            bin_size=(250, 50),
                                                            minimum_track_density=2,
                                                            use_duts=None,
                                                            cut_distance=500,
                                                            max_distance=500,
                                                            col_range=(1250, 17500),
                                                            row_range=(1000, 16000))
        self.assertAlmostEqual(efficiencies[0], 100.0, msg='DUT 0 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[1], 98.701298701298697, msg='DUT 1 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[2], 97.468354430379748, msg='DUT 2 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[3], 100.0, msg='DUT 3 efficiencies do not match')

if __name__ == '__main__':
    tests_data_folder = r'test_result_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResultAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
