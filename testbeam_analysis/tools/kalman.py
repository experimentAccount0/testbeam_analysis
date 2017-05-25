import numpy as np

from numba import njit
from numpy import linalg


@njit
def _filter_predict(transition_matrix, transition_covariance,
                    transition_offset, current_filtered_state,
                    current_filtered_state_covariance):
    """Calculates the predicted state and its covariance matrix. Prediction
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    transition_matrix : [chunk_size, n_dim_state, n_dim_state] array
        state transition matrix from time t to t+1.
    transition_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix for state transition from time t to t+1.
    transition_offset : [chunk_size, n_dim_state] array
        offset for state transition from time t to t+1.
    current_filtered_state: [chunk_size, n_dim_state] array
        filtered state at time t.
    current_filtered_state_covariance: [chunk_size, n_dim_state, n_dim_state] array
        covariance of filtered state at time t.

    Returns
    -------
    predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t+1.
    predicted_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of predicted state at time t+1.
    """
    predicted_state = _vec_mul(transition_matrix, current_filtered_state) + transition_offset

    predicted_state_covariance = _mat_mul(transition_matrix,
                                          _mat_mul(current_filtered_state_covariance,
                                                   _mat_trans(transition_matrix))) + transition_covariance

    return predicted_state, predicted_state_covariance


def _filter_correct(observation_matrix, observation_covariance,
                    observation_offset, predicted_state,
                    predicted_state_covariance, observation, mask):
    r"""Filters a predicted state with the Kalman Filter. Filtering
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    observation_matrix : [chunk_size, n_dim_obs, n_dim_obs] array
        observation matrix for time t.
    observation_covariance : [chunk_size, n_dim_obs, n_dim_obs] array
        covariance matrix for observation at time t.
    observation_offset : [chunk_size, n_dim_obs] array
        offset for observation at time t.
    predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t.
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance matrix of predicted state at time t.
    observation : [chunk_size, n_dim_obs] array
        observation at time t.  If observation is a masked array and any of
        its values are masked, the observation will be not included in filtering.
    mask : [chunk_size, n_dim_obs] bool
        Mask which determines if measurement will be included in filtering step (False, not masked)
        or will be treated as missing measurement (True, masked).

    Returns
    -------
    kalman_gain : [chunk_size, n_dim_state, n_dim_obs] array
        Kalman gain matrix for time t.
    filtered_state : [chunk_size, n_dim_state] array
        filtered state at time t.
    filtered_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t.
    """
    if not np.any(mask):
        predicted_observation = _vec_mul(observation_matrix, predicted_state) + observation_offset

        predicted_observation_covariance = _mat_mul(observation_matrix,
                                                    _mat_mul(predicted_state_covariance, _mat_trans(observation_matrix))) + observation_covariance

        kalman_gain = _mat_mul(predicted_state_covariance,
                               _mat_mul(_mat_trans(observation_matrix),
                                        _mat_inverse(predicted_observation_covariance)))

        filtered_state = predicted_state + _vec_mul(kalman_gain,
                                                    observation - predicted_observation)

        filtered_state_covariance = predicted_state_covariance - _mat_mul(kalman_gain,
                                                                          _mat_mul(observation_matrix,
                                                                                   predicted_state_covariance))
    else:
        n_dim_state = predicted_state_covariance.shape[1]
        n_dim_obs = observation_matrix.shape[1]
        chunk_size = observation_matrix.shape[0]
        kalman_gain = np.zeros((chunk_size, n_dim_state, n_dim_obs))

        filtered_state = predicted_state
        filtered_state_covariance = predicted_state_covariance

    return kalman_gain, filtered_state, filtered_state_covariance


def _filter(transition_matrices, observation_matrices, transition_covariances,
            observation_covariances, transition_offsets, observation_offsets,
            initial_state, initial_state_covariance, observations, mask):
    """Apply the Kalman Filter. First a prediction of the state is done, then a filtering is
    done which includes the observations.

    Parameters
    ----------
    transition_matrices : [chunk_size, n_timesteps-1, n_dim_state, n_dim_state] array-like
        matrices to transport states from t to t+1.
    observation_matrices : [chunk_size, n_timesteps, n_dim_obs, n_dim_state] array-like
        observation matrices.
    transition_covariances : [chunk_size, n_timesteps-1, n_dim_state,n_dim_state]  array-like
        covariance matrices of transition matrices.
    observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
        covariance matrices of observation matrices.
    transition_offsets : [chunk_size, n_timesteps-1, n_dim_state] array-like
        offsets of transition matrices.
    observation_offsets : [chunk_size, n_timesteps, n_dim_obs] array-like
        offsets of observations.
    initial_state : [chunk_size, n_dim_state] array-like
        initial value of state.
    initial_state_covariance : [chunk_size, n_dim_state, n_dim_state] array-like
        initial value for observation covariance matrices.
    observations : [chunk_size, n_timesteps, n_dim_obs] array
        observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
        then observations[:, t] will be treated as a missing observation
        and will not be included in the filtering step.

    Returns
    -------
    predicted_states : [chunk_size, n_timesteps, n_dim_state] array
        predicted states of times [0...t].
    predicted_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
        covariance matrices of predicted states of times [0...t].
    kalman_gains : [chunk_size, n_timesteps, n_dim_state] array
        Kalman gain matrices of times [0...t].
    filtered_states : [chunk_size, n_timesteps, n_dim_state] array
        filtered states of times [0...t].
    filtered_state_covariances : [chunk_size, n_timesteps, n_dim_state] array
        covariance matrices of filtered states of times [0...t].
    """
    chunk_size, n_timesteps, n_dim_obs = observations.shape
    n_dim_state = transition_covariances.shape[2]

    predicted_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_obs))
    filtered_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))

    for t in range(n_timesteps):
        if t == 0:
            predicted_states[:, t] = initial_state
            predicted_state_covariances[:, t] = initial_state_covariance
        else:
            transition_matrix = transition_matrices[:, t - 1]
            transition_covariance = transition_covariances[:, t - 1]
            transition_offset = transition_offsets[:, t - 1]
            predicted_states[:, t], predicted_state_covariances[:, t] = _filter_predict(
                transition_matrix,
                transition_covariance,
                transition_offset,
                filtered_states[:, t - 1],
                filtered_state_covariances[:, t - 1])

        observation_matrix = observation_matrices[:, t]
        observation_covariance = observation_covariances[:, t]
        observation_offset = observation_offsets[:, t]
        kalman_gains[:, t], filtered_states[:, t], filtered_state_covariances[:, t] = _filter_correct(
            observation_matrix,
            observation_covariance,
            observation_offset,
            predicted_states[:, t],
            predicted_state_covariances[:, t],
            observations[:, t],
            np.ma.getmask(observations[:, t]))

    return predicted_states, predicted_state_covariances, kalman_gains, filtered_states, filtered_state_covariances


@njit
def _smooth_update(transition_matrix, filtered_state,
                   filtered_state_covariance, predicted_state,
                   predicted_state_covariance, next_smoothed_state,
                   next_smoothed_state_covariance):
    """Smooth a filtered state with a Kalman Smoother. Smoothing
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    transition_matrix : [chunk_size, n_dim_state, n_dim_state] array
        transition matrix to transport state from time t to t+1.
    filtered_state : [chunk_size, n_dim_state] array
        filtered state at time t.
    filtered_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t.
    predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t+1.
    predicted_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t+1.
    next_smoothed_state : [chunk_size, n_dim_state] array
        smoothed state at time t+1.
    next_smoothed_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of smoothed state at time t+1.

    Returns
    -------
    smoothed_state : [chunk_size, n_dim_state] array
        smoothed state at time t.
    smoothed_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of smoothed state at time t.
    kalman_smoothing_gain : [chunk_size, n_dim_state, n_dim_state] array
        smoothed Kalman gain matrix at time t.
    """
    kalman_smoothing_gain = _mat_mul(filtered_state_covariance,
                                     _mat_mul(_mat_trans(transition_matrix),
                                              _mat_inverse(predicted_state_covariance)))

    smoothed_state = filtered_state + _vec_mul(kalman_smoothing_gain,
                                               next_smoothed_state - predicted_state)

    smoothed_state_covariance = filtered_state_covariance + _mat_mul(kalman_smoothing_gain,
                                                                     _mat_mul((next_smoothed_state_covariance - predicted_state_covariance),
                                                                              _mat_trans(kalman_smoothing_gain)))
    return smoothed_state, smoothed_state_covariance, kalman_smoothing_gain


@njit
def _smooth(transition_matrices, filtered_states,
            filtered_state_covariances, predicted_states,
            predicted_state_covariances):
    """Apply the Kalman Smoother to filtered states. Estimate the smoothed states.
    Smoothing is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    transition_matrices : [chunk_size, n_timesteps-1, n_dim_state, n_dim_state] array-like
        matrices to transport states from t to t+1 of times [0...t-1].
    filtered_states : [chunk_size, n_timesteps, n_dim_state] array
        filtered states of times [0...t].
    filtered_state_covariances : [chunk_size, n_timesteps, n_dim_state] array
        covariance matrices of filtered states of times [0...t].
    predicted_states : [chunk_size, n_timesteps, n_dim_state] array
        predicted states of times [0...t].
    predicted_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
        covariance matrices of predicted states of times [0...t].

    Returns
    -------
    smoothed_states : [chunk_size, n_timesteps, n_dim_state]
        smoothed states for times [0...n_timesteps-1].
    smoothed_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
        covariance matrices of smoothed states for times [0...n_timesteps-1].
    kalman_smoothing_gains : [chunk_size, n_timesteps-1, n_dim_state] array
        smoothed kalman gain matrices fot times [0...n_timesteps-2].
    """
    chunk_size, n_timesteps, n_dim_state = filtered_states.shape
    smoothed_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    smoothed_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    kalman_smoothing_gains = np.zeros((chunk_size, n_timesteps - 1, n_dim_state, n_dim_state))

    smoothed_states[:, -1] = filtered_states[:, -1]
    smoothed_state_covariances[:, -1] = filtered_state_covariances[:, -1]

    for i in range(n_timesteps - 1):
        t = (n_timesteps - 2) - i  # reverse order
        transition_matrix = transition_matrices[:, t]
        smoothed_states[:, t], smoothed_state_covariances[:, t], kalman_smoothing_gains[:, t] = _smooth_update(
            transition_matrix,
            filtered_states[:, t],
            filtered_state_covariances[:, t],
            predicted_states[:, t + 1],
            predicted_state_covariances[:, t + 1],
            smoothed_states[:, t + 1],
            smoothed_state_covariances[:, t + 1])

    return smoothed_states, smoothed_state_covariances, kalman_smoothing_gains


@njit
def _mat_mul(X, Y):
    '''Helper function to multiply two 3D matrices. Multiplication is done on last two axes.
    '''
    result = np.zeros(X.shape)
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            # iterate through columns of Y
            for j in range(Y.shape[2]):
                # iterate through rows of Y
                for k in range(Y.shape[1]):
                    result[l][i][j] += X[l][i][k] * Y[l][k][j]
    return result


@njit
def _vec_mul(X, Y):
    '''Helper function to multiply 3D matrix with 3D vector. Multiplication is done on last two axes.
    '''
    result = np.zeros(Y.shape)
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            # iterate through columns of Y
            for k in range(Y.shape[1]):
                result[l][i] += X[l][i][k] * Y[l][k]
    return result


@njit
def _mat_trans(X):
    '''Helper function to calculate transpose of 3D matrix. Transposition is done on last two axes.
    '''
    result = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for l in range(X.shape[0]):
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                result[l][i][j] = X[l][j][i]

    return result


@njit
def _mat_inverse(X):
    '''Helper function to calculate inverese of 3D matrix. Inversion is done on last two axes.
    '''
    inv = np.zeros((X.shape))
    for i in range(X.shape[0]):
            inv[i] = linalg.pinv(X[i])
    return inv


class KalmanFilter():
    def smooth(self, transition_matrices, transition_offsets, transition_covariance,
               observation_matrices, observation_offsets, observation_covariances,
               initial_state, initial_state_covariance, observations):
        """Apply the Kalman Smoother to the observations. In the first step a filtering is done,
        afterwards a smoothing is done. Calculation is done on whole track chunk with size chunk_size.

        Parameters
        ----------
        transition_matrices : [chunk_size, n_timesteps-1, n_dim_state, n_dim_state] array-like
            matrices to transport states from t to t+1.
        transition_offsets : [chunk_size, n_timesteps-1, n_dim_state] array-like
            offsets of transition matrices.
        transition_covariances : [chunk_size, n_timesteps-1, n_dim_state,n_dim_state]  array-like
            covariance matrices of transition matrices.
        observation_matrices : [chunk_size, n_timesteps, n_dim_obs, n_dim_state] array-like
            observation matrices.
        observation_offsets : [chunk_size, n_timesteps, n_dim_obs] array-like
            offsets of observations.
        observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
            covariance matrices of observation matrices.
        initial_state : [chunk_size, n_dim_state] array-like
            initial value of state.
        initial_state_covariance : [chunk_size, n_dim_state, n_dim_state] array-like
            initial value for observation covariance matrices.
        observations : [chunk_size, n_timesteps, n_dim_obs] array
            observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
            then observations[:, t] will be treated as a missing observation
            and will not be included in the filtering step.

        Returns
        -------
        smoothed_states : [chunk_size, n_timesteps, n_dim_state]
            smoothed states for times [0...n_timesteps-1].
        smoothed_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
            covariance matrices of smoothed states for times [0...n_timesteps-1].
        """
        predicted_states, predicted_state_covariances, _, filtered_states, filtered_state_covariances = _filter(
            transition_matrices, observation_matrices,
            transition_covariance, observation_covariances,
            transition_offsets, observation_offsets,
            initial_state, initial_state_covariance, observations,
            observations.mask)

        smoothed_states, smoothed_state_covariances, _ = _smooth(
            transition_matrices, filtered_states,
            filtered_state_covariances, predicted_states,
            predicted_state_covariances)

        return smoothed_states, smoothed_state_covariances
