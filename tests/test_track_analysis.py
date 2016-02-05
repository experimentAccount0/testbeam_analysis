''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import unittest
import os

from testbeam_analysis import track_analysis
from testbeam_analysis.tools import test_tools

tests_data_folder = r'tests/test_track_analysis/'


class TestTrackAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            from xvfbwrapper import Xvfb
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.output_folder = tests_data_folder
        cls.pixel_size = (250, 50)  # in um
        cls.z_positions = [0., 1.95, 10.88, 12.83]

    @classmethod
    def tearDownClass(cls):  # remove created files
        os.remove(cls.output_folder + 'TrackCandidates.h5')
        os.remove(cls.output_folder + 'Tracks.h5')
        os.remove(cls.output_folder + 'Tracks.pdf')

    def test_track_finding(self):
        track_analysis.find_tracks(tracklets_file=tests_data_folder + 'Tracklets_small.h5',
                                   alignment_file=tests_data_folder + r'Alignment_result.h5',
                                   track_candidates_file=self.output_folder + 'TrackCandidates.h5')
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'TrackCandidates_result.h5', self.output_folder + 'TrackCandidates.h5')
        self.assertTrue(data_equal, msg=error_msg)

    def test_track_fitting(self):
        # Fit the track candidates and create new track table
        track_analysis.fit_tracks(track_candidates_file=tests_data_folder + 'TrackCandidates_result.h5',
                                  tracks_file=self.output_folder + 'Tracks.h5',
                                  output_pdf=self.output_folder + 'Tracks.pdf',
                                  z_positions=self.z_positions,
                                  fit_duts=None,
                                  include_duts=[-3, -2, -1, 1, 2, 3],
                                  ignore_duts=None,
                                  track_quality=1,
                                  use_correlated=False)
        data_equal, error_msg = test_tools.compare_h5_files(tests_data_folder + 'Tracks_result.h5', self.output_folder + 'Tracks.h5', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

if __name__ == '__main__':
    tests_data_folder = r'test_track_analysis/'
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
