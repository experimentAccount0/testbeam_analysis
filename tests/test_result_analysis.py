''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import tables as tb
import numpy as np
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
        cls.z_positions = [0., 1.95, 10.88, 12.83]  # in cm

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass
#         os.remove(cls.output_folder + 'Efficiency.pdf')
#         os.remove(cls.output_folder + 'Residuals.pdf')

    @unittest.SkipTest
    def test_residuals_calculation(self):
        residuals = result_analysis.calculate_residuals(tracks_file=tests_data_folder + 'Tracks_result.h5',
                                                        output_pdf=self.output_folder + 'Residuals.pdf',
                                                        z_positions=self.z_positions,
                                                        use_duts=None,
                                                        max_chi2=3e3)

        # Only test row residuals, columns are to large (250 um) for meaningfull gaussian residuals distribution
        self.assertAlmostEqual(residuals[1], 31.012493732525773, msg='DUT 0 row residuals do not match')
        self.assertAlmostEqual(residuals[3], 21.480965949719472, msg='DUT 1 row residuals do not match')
        self.assertAlmostEqual(residuals[5], 31.372070715636198, msg='DUT 2 row residuals do not match')
        self.assertAlmostEqual(residuals[7], 44.449772465251371, msg='DUT 3 row residuals do not match')

    @unittest.SkipTest
    def test_efficiency_calculation(self):
        efficiencies = result_analysis.calculate_efficiency(tracks_file=self.output_folder + 'Tracks_result.h5',
                                                            output_pdf=self.output_folder + 'Efficiency.pdf',
                                                            z_positions=self.z_positions,
                                                            dim_x=80,
                                                            dim_y=336,
                                                            minimum_track_density=2,
                                                            pixel_size=self.pixel_size,
                                                            use_duts=None,
                                                            cut_distance=500,
                                                            max_distance=500,
                                                            col_range=(5, 70),
                                                            row_range=(20, 320))
        self.assertAlmostEqual(efficiencies[0], 100, msg='DUT 0 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[1], 98.113207547169807, msg='DUT 1 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[2], 97.484276729559738, msg='DUT 2 efficiencies do not match')
        self.assertAlmostEqual(efficiencies[3], 92.666666666666671, msg='DUT 3 efficiencies do not match')

if __name__ == '__main__':
    tests_data_folder = r'test_result_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResultAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
