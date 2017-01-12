from __future__ import division

import logging
from math import sqrt

import tables as tb
import numpy as np

from math import sin
from math import asin


def get_plane_normal(direction_vector_1, direction_vector_2):
    ''' Calculates the normal vector of a plane from two non parallel
    direction vectors within that plane.

    Paramter:
    --------

    direction_vector_1 : array like with 3 dimensions
    direction_vector_2 : array like with 3 dimensions


    Returns:
    --------

    array like with 3 dimension
    '''

    return np.cross(direction_vector_1, direction_vector_2)


def get_line_intersections_with_plane(line_origins, line_directions, position_plane, normal_plane):
    ''' Calculates the intersection of n lines with one plane (n >= 1).
    If there is not an intersection point (line is parallel to plane or the line is in the plane)
    the intersection point is set to NaN.

    Link: http://stackoverflow.com/questions/4938332/line-plane-intersection-based-on-points

    Paramter:
    --------

    line_origins : array like with n, 3 dimensions
        A point of the line for n lines
    line_directions : array like with n, 3 dimensions
        The direction vector of the line for n lines
    position_plane : array like with 3 dimensions
        A vector to the plane
    normal_plane : array like with 3 dimensions
        The normal vector of the plane


    Returns:
    --------

    array like with n, 3 dimension with the intersection point
    '''

    offsets = position_plane[np.newaxis, :] - line_origins  # Calculate offsets and extend in missing dimension

    # Precalculate to be able to avoid division by 0 (line is parallel to the plane or in the plane)
    normal_dot_offsets = np.dot(normal_plane, offsets.T)
    normal_dot_directions = np.atleast_1d(np.dot(normal_plane, line_directions.T))  # Dot product is transformed to be at least 1D for special n = 1

    # Initialize result to nan
    t = np.empty_like(normal_dot_offsets)
    t[:] = np.NAN

    # Warn if some intersection cannot be calculated
    if np.any(normal_dot_directions == 0):
        logging.warning('Some line plane intersection could not be calculated')

    # Calculate t scalar for each line simultaniously, avoid division by 0
    t[normal_dot_directions != 0] = normal_dot_offsets[normal_dot_directions != 0] / normal_dot_directions[normal_dot_directions != 0]

    # Calculate the intersections for each line with the plane
    intersections = line_origins + line_directions * t[:, np.newaxis]

    return intersections


def cartesian_to_spherical(x, y, z):
    ''' Does a transformation from cartesian to spherical coordinates.

    Convention: r = 0 --> phi = theta = 0

    Paramter:
    --------

    x, y, z : number
        Position in cartesian space

    Returns:
    --------

    spherical coordinates: phi, theta, r
    '''

    r = np.sqrt(x * x + y * y + z * z)
    phi = np.zeros_like(r)  # define phi = 0 for x = 0
    theta = np.zeros_like(r)  # theta = 0 for r = 0
    # Avoid division by zero
    phi[x != 0] = np.arctan2(y[x != 0], x[x != 0])  # https://en.wikipedia.org/wiki/Atan2
    phi[phi < 0] += 2. * np.pi  # map to phi = [0 .. 2 pi[
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])
    return phi, theta, r


def spherical_to_cartesian(phi, theta, r):
    ''' Does a transformation from spherical to cartesian coordinates and does error checks.

    Paramter:
    --------

    phi, theta, r : number
        Position in spherical space

    Returns:
    --------

    cartesian coordinates: x, y, z
    '''
    if np.any(r < 0):
        raise RuntimeError('Conversion from spherical to cartesian coordinates failed, because r < 0')
    if np.any(theta < 0) or np.any(theta >= np.pi):
        raise RuntimeError('Conversion from spherical to cartesian coordinates failed, because theta exceeds [0, Pi[')
    if np.any(phi < 0) or np.any(phi >= 2 * np.pi):
        raise RuntimeError('Conversion from spherical to cartesian coordinates failed, because phi exceeds [0, 2*Pi[')
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z


def rotation_matrix_x(angle):
    ''' Calculates the rotation matrix for the rotation around the x axis by an angle
    in a cartesian right-handed coordinate system

    Paramter:
    --------

    angle : number
        Angle in radians

    Returns:
    --------

    np.array with shape 3, 3
    '''

    return np.array([[1, 0, 0],
                     [0, np.cos(angle), np.sin(angle)],
                     [0, -np.sin(angle), np.cos(angle)]])


def rotation_matrix_y(angle):
    ''' Calculates the rotation matrix for the rotation around the y axis by an angle
    in a cartesian right-handed coordinate system

    Paramter:
    --------

    angle : number
        Angle in radians

    Returns:
    --------

    np.array with shape 3, 3
    '''
    return np.array([[np.cos(angle), 0, - np.sin(angle)],
                     [0, 1, 0],
                     [np.sin(angle), 0, np.cos(angle)]])


def rotation_matrix_z(angle):
    ''' Calculates the rotation matrix for the rotation around the z axis by an angle
    in a cartesian right-handed coordinate system

    Paramter:
    --------

    gamma : number
        Angle in radians

    Returns:
    --------

    np.array with shape 3, 3
    '''
    return np.array([[np.cos(angle), np.sin(angle), 0],
                     [-np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])


def rotation_matrix(alpha, beta, gamma):
    ''' Calculates the rotation matrix for the rotation around the three cartesian axis x, y, z
    in a right-handed system. The rotation is done around x then y then z.

    Remember:
        - Transform to the locale coordinate system before applying rotations
        - Rotations are associative but not commutative

    Usage:
    ------
        A rotation by (alpha, beta, gamma) of the vector (x, y, z) in the local coordinate system can be done by:
          np.dot(rotation_matrix(alpha, beta, gamma), np.array([x, y, z]))


    Paramter:
    --------

    alpha : number
        Angle in radians for rotation around x
    beta : number
        Angle in radians for rotation around y
    gamma : number
        Angle in radians for rotation around z

    Returns:
    --------

    np.array with shape 3, 3
    '''

    return np.dot(rotation_matrix_x(alpha), np.dot(rotation_matrix_y(beta), rotation_matrix_z(gamma)))


def translation_matrix(x, y, z):
    ''' Calculates the translation matrix for the translation in x, y, z
    in a cartesian right-handed system.

    Remember:
        - Translations are associative and commutative

    Usage:
    ------
        A translation of a vector (x, y, z) by dx, dy, dz can be done by:
          np.dot(translation_matrix(dx, dy, dz), np.array([x, y, z, 1]))


    Paramter:
    --------

    x : number
        Translation in x
    y : number
        Translation in y
    z : number
        Translation in z

    Returns:
    --------

    np.array with shape 4, 4
    '''

    translation_matrix = np.eye(4, 4, 0)
    translation_matrix[3, :3] = np.array([x, y, z])

    return translation_matrix.T


def global_to_local_transformation_matrix(x, y, z, alpha, beta, gamma):
    ''' Calculates the transformation matrix that applies a translation by T=(-x, -y, -z)
    to the local coordinate system followed by a rotation = R(alpha, beta, gamma).T
    in the local coordinate system.

    This function is the inverse of local_to_global_transformation_matrix()

    Remember:
        - The resulting transformation matrix is 4 x 4
        - Translation and Rotation operations are not commutative

    Paramter:
    --------

    x : number
        Translation in x
    y : number
        Translation in y
    z : number
        Translation in z
    alpha : number
        Angle in radians for rotation around x
    beta : number
        Angle in radians for rotation around y
    gamma : number
        Angle in radians for rotation around z

    Returns:
    --------
    np.array with shape 4, 4
    '''

    # Extend rotation matrix R by one dimension
    R = np.eye(4, 4, 0)
    # Inverse of rotation Matrix Rtot_inv = R(alpha) * R(beta) * R(gamma) = R(gamma).T * R(beta).T * R(alpha).T = (R(alpha) * R(beta) * R(gamma)).T = Rtot.T
    R[:3, :3] = rotation_matrix(alpha, beta, gamma).T  # Inverse of a rotation matrix is also the transformed matrix, since Det = 1

    # Get translation matrix T
    T = translation_matrix(-x, -y, -z)

    return np.dot(R, T)


def local_to_global_transformation_matrix(x, y, z, alpha, beta, gamma):
    ''' Calculates the transformation matrix that applies an inverse rotation in the local coordinate system
    followed by an inverse translation by x, y, z to the global coordinate system.

    Remember:
        - The resulting transformation matrix is 4 x 4
        - Translation and Rotation operations do not commutative

    Paramter:
    --------

    x : number
        Translation in x
    y : number
        Translation in y
    z : number
        Translation in z
    alpha : number
        Angle in radians for rotation around x
    beta : number
        Angle in radians for rotation around y
    gamma : number
        Angle in radians for rotation around z

    Returns:
    --------
    np.array with shape 4, 4
    '''

    # Extend inverse rotation matrix R by one dimension
    R = np.eye(4, 4, 0)
    R[:3, :3] = rotation_matrix(alpha, beta, gamma)

    # Get inverse translation matrix T
    T = translation_matrix(x, y, z)

    return np.dot(T, R)


def apply_transformation_matrix(x, y, z, transformation_matrix):
    ''' Takes array in x, y, z and applies a transformation matrix (4 x 4).

    Paramter:
    --------

    x : number
        Position in x
    y : number
        Position in y
    z : number
        Position in z

    Returns:
    --------
    np.array with shape 3, 3
    '''

    positions = np.column_stack((x, y, z, np.ones_like(x))).T  # Add extra 4th dimension
    positions_transformed = np.dot(transformation_matrix, positions).T[:, :-1]  # Transform and delete extra dimension

    return positions_transformed[:, 0], positions_transformed[:, 1], positions_transformed[:, 2]


def apply_rotation_matrix(x, y, z, rotation_matrix):
    ''' Takes array in x, y, z and applies a rotation matrix (3 x 3).

    Paramter:
    --------

    x : number
        Position in x
    y : number
        Position in y
    z : number
        Position in z

    Returns:
    --------
    np.array with shape 3, 3
    '''

    positions = np.column_stack((x, y, z)).T
    positions_transformed = np.dot(rotation_matrix, positions).T

    return positions_transformed[:, 0], positions_transformed[:, 1], positions_transformed[:, 2]


def apply_alignment(hits_x, hits_y, hits_z, dut_index, alignment=None, prealignment=None, inverse=False):
    ''' Helper function that takes hits and applies a transformation according to the alignment data given.
    If alignment data with rotations and translations are given the hits are transformed according to the
    rotations and translations.
    If pre-alignment data with offsets and slopes are given the hits are transformed according to the
    slopes and offsets.
    If both are given alignment data is taken.
    The transformation can be inverted.

    Paramter:
    --------

    hits_x, hits_y, hits_z : numpy arrays with corresponding hit positions
    dut_index : integer
        Needed to select the corrct alignment info
    alignment : nunmpy array
        Alignment information with rotations and translations
    prealignment : numpy array
        Pre-alignment information with offsets and slopes
    inverse : boolean
        Apply inverse transformation if true

    Returns:
    --------
    hits_x, hits_y, hits_z : numpy arrays
    '''
    if (alignment is None and prealignment is None) or (alignment is not None and prealignment is not None):
        raise RuntimeError('Either pre-alignment or alignment data has to be given.')

    if alignment is not None:
        if inverse:
            logging.debug('Transform hit position into the local coordinate system using alignment data')
            transformation_matrix = global_to_local_transformation_matrix(x=alignment[dut_index]['translation_x'],
                                                                          y=alignment[dut_index]['translation_y'],
                                                                          z=alignment[dut_index]['translation_z'],
                                                                          alpha=alignment[dut_index]['alpha'],
                                                                          beta=alignment[dut_index]['beta'],
                                                                          gamma=alignment[dut_index]['gamma'])
        else:
            logging.debug('Transform hit position into the global coordinate system using alignment data')
            transformation_matrix = local_to_global_transformation_matrix(x=alignment[dut_index]['translation_x'],
                                                                          y=alignment[dut_index]['translation_y'],
                                                                          z=alignment[dut_index]['translation_z'],
                                                                          alpha=alignment[dut_index]['alpha'],
                                                                          beta=alignment[dut_index]['beta'],
                                                                          gamma=alignment[dut_index]['gamma'])

        hits_x, hits_y, hits_z = apply_transformation_matrix(x=hits_x,
                                                             y=hits_y,
                                                             z=hits_z,
                                                             transformation_matrix=transformation_matrix)
    else:
        c0_column = prealignment[dut_index]['column_c0']
        c1_column = prealignment[dut_index]['column_c1']
        c0_row = prealignment[dut_index]['row_c0']
        c1_row = prealignment[dut_index]['row_c1']
        z = prealignment[dut_index]['z']

        if inverse:
            logging.debug('Transform hit position into the local coordinate system using pre-alignment data')
            hits_x = (hits_x - c0_column) / c1_column
            hits_y = (hits_y - c0_row) / c1_row
            hits_z -= z
        else:
            logging.debug('Transform hit position into the global coordinate system using pre-alignment data')
            hits_x = (c1_column * hits_x + c0_column)
            hits_y = (c1_row * hits_y + c0_row)
            hits_z += z

    return hits_x, hits_y, hits_z


def merge_alignment_parameters(old_alignment, new_alignment, mode='relative', select_duts=None):
    if select_duts is None:  # select all DUTs
        dut_selection = np.ones(old_alignment.shape[0], dtype=np.bool)
    else:
        dut_selection = np.zeros(old_alignment.shape[0], dtype=np.bool)
        dut_selection[np.array(select_duts)] = True

    alignment_parameters = old_alignment.copy()  # Do not change input parameters

    if mode == 'absolute':
        logging.info('Set alignment')
        alignment_parameters[dut_selection] = new_alignment[dut_selection]
        return alignment_parameters
    elif mode == 'relative':
        logging.info('Merge new alignment with old alignment')

        alignment_parameters['translation_x'][dut_selection] += new_alignment['translation_x'][dut_selection]
        alignment_parameters['translation_y'][dut_selection] += new_alignment['translation_y'][dut_selection]
        alignment_parameters['translation_z'][dut_selection] += new_alignment['translation_z'][dut_selection]

        alignment_parameters['alpha'][dut_selection] += new_alignment['alpha'][dut_selection]
        alignment_parameters['beta'][dut_selection] += new_alignment['beta'][dut_selection]
        alignment_parameters['gamma'][dut_selection] += new_alignment['gamma'][dut_selection]

        # TODO: Is this always a good idea? Usually works, but what if one heavily tilted device?
        # All alignments are relative, thus center them around 0 by substracting the mean (exception: z position)
        if np.count_nonzero(dut_selection) > 1:
            alignment_parameters['alpha'][dut_selection] -= np.mean(alignment_parameters['alpha'][dut_selection])
            alignment_parameters['beta'][dut_selection] -= np.mean(alignment_parameters['beta'][dut_selection])
            alignment_parameters['gamma'][dut_selection] -= np.mean(alignment_parameters['gamma'][dut_selection])
            alignment_parameters['translation_x'][dut_selection] -= np.mean(alignment_parameters['translation_x'][dut_selection])
            alignment_parameters['translation_y'][dut_selection] -= np.mean(alignment_parameters['translation_y'][dut_selection])

        return alignment_parameters
    else:
        raise RuntimeError('Unknown mode %s', str(mode))


def store_alignment_parameters(alignment_file, alignment_parameters, mode='absolute', select_duts=None):
    ''' Stores the alignment parameters (rotations, translations) into the alignment file.
    Absolute (overwriting) and relative mode (add angles, translations) is supported.

    Paramter:
    --------

    alignment_file : pytables file
        The pytables file with the alignment
    alignment_parameters : numpy recarray
        An array with the alignment values
    mode : string
        'relative' and 'absolute' supported
    use_duts : iterable
        In relative mode only change specified DUTs
    '''

    with tb.open_file(alignment_file, mode="r+") as out_file_h5:  # Open file with alignment data
        #FIXME: this does not make sence to be here: 
#         alignment_parameters[:]['translation_z'] = out_file_h5.root.PreAlignment[:]['z']  # Set z from pre-alignment
        try:
            alignment_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', title='Table containing the alignment geometry parameters (translations and rotations)', description=np.zeros((1,), dtype=alignment_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            alignment_table.append(alignment_parameters)
        except tb.NodeError:
            alignment_parameters = merge_alignment_parameters(old_alignment=out_file_h5.root.Alignment[:],
                                                              new_alignment=alignment_parameters,
                                                              mode=mode,
                                                              select_duts=select_duts)

            logging.info('Overwrite existing alignment!')
            out_file_h5.root.Alignment._f_remove()  # Remove old node, is there a better way?
            alignment_table = out_file_h5.create_table(out_file_h5.root, name='Alignment', title='Table containing the alignment geometry parameters (translations and rotations)', description=np.zeros((1,), dtype=alignment_parameters.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            alignment_table.append(alignment_parameters)

        string = "\n".join(['DUT%d: alpha=%1.4f, beta=%1.4f, gamma=%1.4f Rad, x/y/z=%d/%d/%d um' % (dut_values['DUT'],
                                                                                                dut_values['alpha'],
                                                                                                dut_values['beta'],
                                                                                                dut_values['gamma'],
                                                                                                dut_values['translation_x'],
                                                                                                dut_values['translation_y'],
                                                                                                dut_values['translation_z']) for dut_values in alignment_parameters])
        logging.info('Set alignment parameters to:\n%s' % string)
