''' Helper functions for geometrical operations.
'''

from __future__ import division

import logging

import tables as tb
import numpy as np


def get_plane_normal(direction_vector_1, direction_vector_2):
    ''' Normal vector of a plane.

    Plane is define by two non parallel direction vectors within the plane.

    Paramter:
    --------
    direction_vector_1 : array like with 3 dimensions
    direction_vector_2 : array like with 3 dimensions


    Returns:
    --------
    array like with 3 dimension
    '''

    return np.cross(direction_vector_1, direction_vector_2)


def get_line_intersections_with_plane(line_origins, line_directions,
                                      position_plane, normal_plane):
    ''' Calculates the intersection of n lines with one plane.

    If there is no intersection point (line is parallel to plane or the line is
    in the plane) the intersection point is set to nan.

    Notes
    -----
    Further information:
    http://stackoverflow.com/questions/4938332/line-plane-intersection-based-on-points

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

    # Calculate offsets and extend in missing dimension
    offsets = position_plane[np.newaxis, :] - line_origins

    # Precalculate to be able to avoid division by 0
    # (line is parallel to the plane or in the plane)
    norm_dot_off = np.dot(normal_plane, offsets.T)
    # Dot product is transformed to be at least 1D for special n = 1
    norm_dot_dir = np.atleast_1d(np.dot(normal_plane,
                                        line_directions.T))

    # Initialize result to nan
    t = np.empty_like(norm_dot_off)
    t[:] = np.NAN

    # Warn if some intersection cannot be calculated
    if np.any(norm_dot_dir == 0):
        logging.warning('Some line plane intersection could not be calculated')

    # Calculate t scalar for each line simultaniously, avoid division by 0
    sel = norm_dot_dir != 0
    t[sel] = norm_dot_off[sel] / norm_dot_dir[sel]

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
    # https://en.wikipedia.org/wiki/Atan2
    phi[x != 0] = np.arctan2(y[x != 0], x[x != 0])
    phi[phi < 0] += 2. * np.pi  # map to phi = [0 .. 2 pi[
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])
    return phi, theta, r


def spherical_to_cartesian(phi, theta, r):
    ''' Transformation from spherical to cartesian coordinates.

    Includes error checks.

    Paramter:
    --------
    phi, theta, r : number
        Position in spherical space

    Returns:
    --------
    Cartesian coordinates: x, y, z
    '''
    if np.any(r < 0):
        raise RuntimeError(
            'Conversion from spherical to cartesian coordinates failed, '
            'because r < 0')
    if np.any(theta < 0) or np.any(theta >= np.pi):
        raise RuntimeError(
            'Conversion from spherical to cartesian coordinates failed, '
            'because theta exceeds [0, Pi[')
    if np.any(phi < 0) or np.any(phi >= 2 * np.pi):
        raise RuntimeError(
            'Conversion from spherical to cartesian coordinates failed, '
            'because phi exceeds [0, 2*Pi[')
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z


def rotation_matrix_x(angle):
    ''' Rotation matrix for rotation around x axis by an angle.

    Note:
    -----
    Rotation in a cartesian right-handed coordinate system

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
    ''' Rotation matrix for rotation around y axis by an angle.

    Note:
    -----
    Rotation in a cartesian right-handed coordinate system


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
    ''' Rotation matrix for rotation around z axis by an angle.

    Note:
    -----
    Rotation in a cartesian right-handed coordinate system


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
    ''' Rotation matrix for the rotation around the three cartesian axis x, y, z.

    Note:
    -----
    In a right-handed system. The rotation is done around x then y then z.

    Remember:
        - Transform to the locale coordinate system before applying rotations
        - Rotations are associative but not commutative

    Usage:
    ------
        A rotation by (alpha, beta, gamma) of the vector (x, y, z) in the local
        coordinate system can be done by:
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

    return np.dot(rotation_matrix_x(alpha),
                  np.dot(rotation_matrix_y(beta), rotation_matrix_z(gamma)))


def translation_matrix(x, y, z):
    ''' Translation matrix for the translation in x, y, z in a cartesian system.

    Note:
    -----
        Remember: Translations are associative and commutative

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
    ''' Transformation matrix that applies a translation and rotation.


    Translation is T=(-x, -y, -z) to the local coordinate system followed
    by a rotation = R(alpha, beta, gamma).T in the local coordinate system.

    Note:
    -----
        - This function is the inverse of
          local_to_global_transformation_matrix()
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
    R[:3, :3] = rotation_matrix(alpha, beta, gamma).T

    # Get translation matrix T
    T = translation_matrix(-x, -y, -z)

    return np.dot(R, T)


def local_to_global_transformation_matrix(x, y, z, alpha, beta, gamma):
    ''' Transformation matrix that applies a inverse translation and rotation.

    Inverse rotation in the local coordinate system followed by an inverse
    translation by x, y, z to the global coordinate system.

    Note:
    -----
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
    ''' Takes arrays for x, y, z and applies a transformation matrix (4 x 4).

    Paramter:
    --------

    x : array
        Position in x
    y : array
        Position in y
    z : array
        Position in z

    Returns:
    --------
    np.array with shape 3, 3
    '''

    # Add extra 4th dimension
    pos = np.column_stack((x, y, z, np.ones_like(x))).T

    # Transform and delete extra dimension
    pos_T = np.dot(transformation_matrix, pos).T[:, :-1]

    return pos_T[:, 0], pos_T[:, 1], pos_T[:, 2]


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

    pos = np.column_stack((x, y, z)).T
    pos_T = np.dot(rotation_matrix, pos).T

    return pos_T[:, 0], pos_T[:, 1], pos_T[:, 2]


def apply_alignment(hits_x, hits_y, hits_z, dut_index, alignment=None,
                    prealignment=None, inverse=False):
    ''' Takes hits and applies a transformation according to the alignment data.

    If alignment data with rotations and translations are given the hits are
    transformed according to the rotations and translations.
    If pre-alignment data with offsets and slopes are given the hits are
    transformed according to the slopes and offsets.
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
    if (alignment is None and prealignment is None) or \
       (alignment is not None and prealignment is not None):
        raise RuntimeError('Neither pre-alignment or alignment data given.')

    if alignment is not None:
        if inverse:
            logging.debug('Transform hit position into the local coordinate '
                          'system using alignment data')
            transformation_matrix = global_to_local_transformation_matrix(
                x=alignment[dut_index]['translation_x'],
                y=alignment[dut_index]['translation_y'],
                z=alignment[dut_index]['translation_z'],
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])
        else:
            logging.debug('Transform hit position into the global coordinate '
                          'system using alignment data')
            transformation_matrix = local_to_global_transformation_matrix(
                x=alignment[dut_index]['translation_x'],
                y=alignment[dut_index]['translation_y'],
                z=alignment[dut_index]['translation_z'],
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])

        hits_x, hits_y, hits_z = apply_transformation_matrix(
            x=hits_x,
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
            logging.debug('Transform hit position into the local coordinate '
                          'system using pre-alignment data')
            hits_x = (hits_x - c0_column) / c1_column
            hits_y = (hits_y - c0_row) / c1_row
            hits_z -= z
        else:
            logging.debug('Transform hit position into the global coordinate '
                          'system using pre-alignment data')
            hits_x = (c1_column * hits_x + c0_column)
            hits_y = (c1_row * hits_y + c0_row)
            hits_z += z

    return hits_x, hits_y, hits_z


def merge_alignment_parameters(old_alignment, new_alignment, mode='relative',
                               select_duts=None):
    if select_duts is None:  # Select all DUTs
        dut_sel = np.ones(old_alignment.shape[0], dtype=np.bool)
    else:
        dut_sel = np.zeros(old_alignment.shape[0], dtype=np.bool)
        dut_sel[np.array(select_duts)] = True

    # Do not change input parameters
    alig_pars = old_alignment.copy()

    if mode == 'absolute':
        logging.info('Set alignment')
        alig_pars[dut_sel] = new_alignment[dut_sel]
        return alig_pars
    elif mode == 'relative':
        logging.info('Merge new alignment with old alignment')

        alig_pars['translation_x'][dut_sel] += new_alignment[
            'translation_x'][dut_sel]
        alig_pars['translation_y'][dut_sel] += new_alignment[
            'translation_y'][dut_sel]
        alig_pars['translation_z'][dut_sel] += new_alignment[
            'translation_z'][dut_sel]

        alig_pars['alpha'][dut_sel] += new_alignment['alpha'][dut_sel]
        alig_pars['beta'][dut_sel] += new_alignment['beta'][dut_sel]
        alig_pars['gamma'][dut_sel] += new_alignment['gamma'][dut_sel]

        # TODO: Is this always a good idea? Usually works, but what if one
        # heavily tilted device?
        # All alignments are relative, thus center them around 0 by
        # substracting the mean (exception: z position)
        if np.count_nonzero(dut_sel) > 1:
            alig_pars['alpha'][dut_sel] -= np.mean(alig_pars['alpha'][dut_sel])
            alig_pars['beta'][dut_sel] -= np.mean(alig_pars['beta'][dut_sel])
            alig_pars['gamma'][dut_sel] -= np.mean(alig_pars['gamma'][dut_sel])
            alig_pars['translation_x'][dut_sel] -= np.mean(alig_pars[
                'translation_x'][dut_sel])
            alig_pars['translation_y'][dut_sel] -= np.mean(alig_pars[
                'translation_y'][dut_sel])

        return alig_pars
    else:
        raise RuntimeError('Unknown mode %s', str(mode))


def store_alignment_parameters(alignment_file, alignment_parameters,
                               mode='absolute', select_duts=None):
    ''' Stores alignment parameters (rotations, translations) into file.

    Absolute (overwriting) and relative (add angles, translations) supported.

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

    description = np.zeros((1,), dtype=alignment_parameters.dtype).dtype

    # Open file with alignment data
    with tb.open_file(alignment_file, mode="r+") as out_file:
        try:
            align_tab = out_file.create_table(out_file.root, name='Alignment',
                                              title='Table containing the '
                                              'alignment geometry parameters '
                                              '(translations and rotations)',
                                              description=description,
                                              filters=tb.Filters(
                                                  complib='blosc',
                                                  complevel=5,
                                                  fletcher32=False))
            align_tab.append(alignment_parameters)
        except tb.NodeError:
            align_pars = merge_alignment_parameters(
                old_alignment=out_file.root.Alignment[:],
                new_alignment=alignment_parameters,
                mode=mode,
                select_duts=select_duts)

            logging.info('Overwrite existing alignment!')
            # Remove old node, is there a better way?
            out_file.root.Alignment._f_remove()
            align_tab = out_file.create_table(out_file.root, name='Alignment',
                                              title='Table containing the '
                                              'alignment geometry parameters '
                                              '(translations and rotations)',
                                              description=description,
                                              filters=tb.Filters(
                                                  complib='blosc',
                                                  complevel=5,
                                                  fletcher32=False))
            align_tab.append(align_pars)

        string = "\n".join(['DUT%d: alpha=%1.4f, beta=%1.4f, gamma=%1.4f Rad, '
                            'x/y/z=%d/%d/%d um' % (dut_values['DUT'],
                                                   dut_values['alpha'],
                                                   dut_values['beta'],
                                                   dut_values['gamma'],
                                                   dut_values['translation_x'],
                                                   dut_values['translation_y'],
                                                   dut_values['translation_z'])
                            for dut_values in align_pars])
        logging.info('Set alignment parameters to:\n%s' % string)
