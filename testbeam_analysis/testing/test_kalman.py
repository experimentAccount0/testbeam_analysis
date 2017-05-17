''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

import numpy as np

from testbeam_analysis import track_analysis
from testbeam_analysis.tools import test_tools
from pykalman.standard import KalmanFilter

# Get package path
testing_path = os.path.dirname(__file__)  # Get the absoulte path of the online_monitor installation

# Set the converter script path
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(testing_path)) + r'/testing/fixtures/track_analysis/'))


def _kalman_fit_3d(hits, dut_fit_selection, transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance, observation_offset, initial_state_mean, initial_state_covariance):
    kf = KalmanFilter(transition_matrix, observation_matrix,
                      transition_covariance, observation_covariance,
                      transition_offset, observation_offset,
                      initial_state_mean, initial_state_covariance)
    measurements = np.ma.array(np.c_[hits[:, 0:2], np.zeros((hits.shape[0])), np.zeros((hits.shape[0]))])  # x and y pos, slopes in y and y are zero

    # mask duts which should not used in fit
    for dut_index in range(0, hits.shape[0]):
        if dut_index not in dut_fit_selection:
            measurements[dut_index] = np.ma.masked

    smoothed_state_estimates, cov = kf.smooth(measurements)

    x_err = np.sqrt(np.diagonal(cov, axis1=2, axis2=1))[:, 0]
    y_err = np.sqrt(np.diagonal(cov, axis1=2, axis2=1))[:, 1]

    chi2 = np.sum(np.square(measurements[:, 0:2] - smoothed_state_estimates[:, 0:2]), dtype=np.uint32)

    return smoothed_state_estimates, chi2, x_err, y_err


def _fit_tracks_kalman_loop(track_hits, dut_fit_selection, pixel_size, z_positions, beam_energy, total_thickness, radiation_length):
    n_duts = track_hits.shape[1]
    dut_selection = np.array(range(0, n_duts))
    z_positions = np.array(z_positions)

    # create a list of duts which should be included in the fit
    dut_list = np.full(shape=(n_duts), fill_value=np.nan)
    for index in range(n_duts):
        dut_n = index
        if np.bitwise_and(1 << index, dut_fit_selection) == 2 ** index:
            dut_list[dut_n] = dut_n
    dut_fit_selection = dut_list[~np.isnan(dut_list)].astype(int)

    # set multiple scattering environment
    total_thickness = np.array(total_thickness)
    radiation_length = np.array(radiation_length)  # total radiation length of 2 kapton foils (before and after sensor) and m26 sensor in um

    pixel_resolution = pixel_size / np.sqrt(12.)  # Resolution of each telescope plane

    # Calculate multiple scattering
    mass = 0.511  # mass in MeV (electrons)
    momentum = np.sqrt(beam_energy * beam_energy - mass * mass)
    beta = momentum / beam_energy  # almost 1
    # rms angle of multiple scattering
    theta = np.array(((13.6 / momentum / beta) * np.sqrt(total_thickness / radiation_length) * (1. + 0.038 * np.log(total_thickness / radiation_length))))

    track_estimates_chunk = np.zeros((track_hits.shape[0], track_hits.shape[1], 4))
    chi2_tot = np.empty((track_hits.shape[0],), dtype=np.float)
    x_errs = np.zeros((track_hits.shape[0], track_hits.shape[1]))
    y_errs = np.zeros((track_hits.shape[0], track_hits.shape[1]))

    # express transition covariance matrix
    transition_covariance = np.zeros((n_duts - 1, 4, 4))

    # express transition matrix
    transition_matrix = np.zeros((n_duts - 1, 4, 4))

    # express state vector
    state_vector = np.zeros((n_duts - 1, 4,))

    # express transition and observation offset matrices
    transition_offset = np.zeros((n_duts - 1, 4))
    observation_offset = np.zeros((n_duts, 4))

    # express observatipon matrix
    observation_matrix = np.zeros((n_duts, 4, 4))

    # express observation covariance matrices
    observation_covariance = np.zeros((n_duts, 4, 4))
    # observation matrix: only observe x and y position
    observation_matrix[:, 0, 0] = 1.
    observation_matrix[:, 1, 1] = 1.
    # express initial state covariance matrices: x and y pos have initial error of pixel resolution and x and y slopes have large error
    initial_state_covariance = np.zeros((4, 4))
    initial_state_covariance[0, 0] = pixel_resolution[0, 0]**2
    initial_state_covariance[1, 1] = pixel_resolution[0, 1]**2
    initial_state_covariance[2, 2] = 0.01
    initial_state_covariance[3, 3] = 0.01

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        x_positions = np.array(actual_hits[:, 0])
        y_positions = np.array(actual_hits[:, 1])

        # initial state: first two components x and y hit position in first dut, slopes all zero
        initial_state_mean = np.array([actual_hits[0, 0], actual_hits[0, 1], 0., 0.])

        # express observation covariance
        observation_covariance[dut_selection, 0, 0] = np.square(pixel_resolution[dut_selection][:, 0])
        observation_covariance[dut_selection, 1, 1] = np.square(pixel_resolution[dut_selection][:, 1])

        # express transition matrices
        transition_matrix[dut_selection[:-1], :, 0] = np.array([1., 0., 0., 0.])
        transition_matrix[dut_selection[:-1], :, 1] = np.array([0., 1., 0., 0.])
        # need to check wether the slope is negative or positive
        transition_matrix[dut_selection[:-1], :, 2] = np.array([-(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]), np.zeros((len(dut_selection[:-1]),)), np.ones((len(dut_selection[:-1]),)), np.zeros((len(dut_selection[:-1]),))]).T
        transition_matrix[dut_selection[:-1], :, 3] = np.array([np.zeros((len(dut_selection[:-1]),)), - (z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]), np.zeros((len(dut_selection[:-1]),)), np.ones((len(dut_selection[:-1]),))]).T

        state_vector[dut_fit_selection[:-1]] = np.array([x_positions[dut_fit_selection[:-1]],
                                                         y_positions[dut_fit_selection[:-1]],
                                                         np.zeros(x_positions[dut_fit_selection[:-1]].shape),
                                                         np.zeros(y_positions[dut_fit_selection[:-1]].shape)]).T

        cov_33 = theta[dut_selection[:-1]]**2 * (1 + state_vector[dut_selection[:-1], 2]**2) * (1 + state_vector[dut_selection[:-1], 2]**2 + state_vector[dut_selection[:-1], 3]**2).T
        cov_44 = theta[dut_selection[:-1]]**2 * (1 + state_vector[dut_selection[:-1], 3]**2) * (1 + state_vector[dut_selection[:-1], 2]**2 + state_vector[dut_selection[:-1], 3]**2).T
        cov_34 = theta[dut_selection[:-1]]**2 * state_vector[dut_selection[:-1], 2] * state_vector[dut_selection[:-1], 3] * (1 + state_vector[dut_selection[:-1], 2]**2 + state_vector[dut_selection[:-1], 3]**2).T

        # express transition covariance matrices, according to http://web-docs.gsi.de/~ikisel/reco/Methods/CovarianceMatrices-NIMA329-1993.pdf
        transition_covariance[dut_selection[:-1], :, 0] = np.array([(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]])**2 * cov_33,
                                                                    (z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]])**2 * cov_34,
                                                                    -(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_33,
                                                                    -(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_34]).T

        transition_covariance[dut_selection[:-1], :, 1] = np.array([(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]])**2 * cov_34,
                                                                    (z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]])**2 * cov_44,
                                                                    -(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_34,
                                                                    -(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_44]).T
        transition_covariance[dut_selection[:-1], :, 2] = np.array([-(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_33,
                                                                    -(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_34,
                                                                    cov_33,
                                                                    cov_34]).T
        transition_covariance[dut_selection[:-1], :, 3] = np.array([-(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_34,
                                                                    -(z_positions[dut_selection[:-1] + 1] - z_positions[dut_selection[:-1]]) * cov_44,
                                                                    cov_34,
                                                                    cov_44]).T
        # run kalman filter
        track_estimate_chunk, chi2, x_err, y_err = _kalman_fit_3d(actual_hits, dut_fit_selection,
                                                                  transition_matrix, transition_covariance,
                                                                  transition_offset, observation_matrix,
                                                                  observation_covariance, observation_offset,
                                                                  initial_state_mean, initial_state_covariance)

        chi2_tot[index] = chi2
        track_estimates_chunk[index] = track_estimate_chunk
        x_errs[index] = x_err
        y_errs[index] = y_err

    return track_estimates_chunk, chi2_tot, x_errs, y_errs


class TestTrackAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.getenv('TRAVIS', False):
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        # cls.output_folder = tests_data_folder
        # cls.pixel_size = (250, 50)  # in um

    @classmethod
    def tearDownClass(cls):  # Remove created files
        pass
        # os.remove(os.path.join(cls.output_folder, 'Tracks_merged.pdf'))

    def test_kalman(self):
        kwargs = {'track_hits': np.array([[[-1229.22372954, 2828.19616302, 0.],
                                           [np.nan, np.nan, np.nan],  # [-1254.51224282, 2827.4291421, 29900.],
                                           [-1285.6117892, 2822.34536687, 60300.],
                                           [-1311.31083616, 2823.56121414, 82100.],
                                           [-1335.8529645, 2828.43359043, 118700.],
                                           [-1357.81872222, 2840.86947964, 160700.],
                                           [-1396.35698339, 2843.76799577, 197800.]]]),
                  'dut_fit_selection': 61,
                  'z_positions': [[0., 29900, 60300, 82100, 118700, 160700, 197800]],
                  'pixel_size': ((18.5, 18.5), (18.5, 18.5), (18.5, 18.5),
                                 (18.5, 18.5), (18.5, 18.5), (250, 50), (250, 50)),
                  'beam_energy': 2500.,
                  'total_thickness': [[100., 100., 100., 100., 100., 100., 250.]],
                  'radiation_length': [[125390., 125390., 125390., 125390., 125390., 125390., 93700.]]}
        for i in range(4):  # test each return (state estimates, chi, x error, y errors) seperatly
            test = test_tools._call_function_with_args(function=track_analysis._fit_tracks_kalman_loop,
                                                       **kwargs)[0][i]
            orig = test_tools._call_function_with_args(function=_fit_tracks_kalman_loop,
                                                       **kwargs)[0][i]
            data_equal = test_tools.array_close(test, orig)
            self.assertTrue(data_equal)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
