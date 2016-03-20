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
    ''' Calculates the intersection of n lines with a plane (n >= 1).
    If there is not a intersection point (line is parallel to plane or the line is in the plane)
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

    array like with n, 3 dimension with the intersection point. If n = 1
    '''

    offsets = position_plane[np.newaxis, :] - line_origins  # Calculate offsets and extend in missing dimension

#     print 'offsets', offsets
#     print 'offsets.shape', offsets.shape
#     print 'normal_plane', normal_plane.shape
#     print 'normal_plane dot offsets', np.dot(normal_plane, offsets.T)
#     print 'direction', line_directions
#     print 'direction.shape', line_directions.shape
#     print 'normal_plane.dot direction', np.dot(normal_plane, line_directions.T)

    # Precalculate to be able to avoid division by 0 (line is parallel to the plane or in the plane)
    normal_dot_offsets = np.dot(normal_plane, offsets.T)
    normal_dot_directions = np.atleast_1d(np.dot(normal_plane, line_directions.T))  # Dot product is transformed to be at least 1D for n = 1

    # Initialize to nan
    t = np.empty_like(normal_dot_offsets)
    t[:] = np.NAN

    # Warn if some intersection cannot be calculated
    if np.any(normal_dot_directions == 0):
        logging.warning('Some line plane intersection could not be calculated')

    # Calculate t scalar for each line simultaniously, avoid division by 0
    t[normal_dot_directions != 0] = normal_dot_offsets[normal_dot_directions != 0] / normal_dot_directions[normal_dot_directions != 0]

    # Calculate the intersections for each line with the plane
    intersections = line_origins + line_directions * t[:, np.newaxis]

    return np.squeeze(intersections)  # Reduce extra dimensions for the n = 1 case


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
          np.dot(rotation_matrix(dx, dy, dz), np.array([x, y, z]))


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
    ''' Takes array in x, y, z and applies a transformation matrix

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


def create_initial_geometry(outFile, zpos, initial_translation=None, initial_rotation=None):
    with tb.open_file(outFile, mode='w') as geoFile:

        nplanes = len(zpos)

        description = [('plane_number', np.int)]
        for index in range(3):
            description.append(('translation_%d' % index, np.float))
        for i in range(3):
            for j in range(3):
                description.append(('rotation_%d_%d' % (i, j), np.float))
        '''description.append(('sin_alpha', np.float))
        description.append(('sin_beta', np.float))
        description.append(('sin_gamma', np.float))'''
        ''' Is it necessary to specify the angles as well?'''

        geo_pars = np.zeros((nplanes,), dtype=description)
        if initial_translation is None:
            xy_translation_t = np.zeros((nplanes, 2))
        else:
            xy_translation_t = initial_translation
        rotation_matrixes = [np.eye(3) for i in range(nplanes)]

        if (initial_translation is not None and len(zpos) == initial_translation.shape[0]) or initial_translation is None:
            xy_translation = np.c_[xy_translation_t, zpos]
        else:
            print("create_initial_geometry: number of planes in initial translation doesn't match the one given by zpos")
            print("create_initial_geometry: setting all translations to 0")

        if initial_rotation is not None:
            if initial_rotation.shape == rotation_matrixes.shape:
                rotation_matrixes = initial_rotation
            else:
                print("create_initial_geometry: initial_rotation must have shape (n_planes,3,3)")
                print("create_initial_geometry: setting all rotations to Identity")

        for index in range(3):
            geo_pars['translation_%d' % index] = xy_translation[:, index]
        for i in range(3):
            for j in range(3):
                geo_pars['rotation_%d_%d' % (i, j)] = [rotation_matrixes[n][i, j] for n in range(nplanes)]

        geo_table = geoFile.create_table(geoFile.root, name='Geometry', title='File containing all the geometry parameters', description=np.zeros((1,), dtype=geo_pars.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        geo_table.append(geo_pars)


def recontruct_geometry_from_file(fileName):
    with tb.open_file(fileName, mode='r') as geoFile:

        geo_pars = geoFile.root.Geometry[:]
        nplanes = geo_pars.shape[0]

        xy_translation = np.zeros((nplanes, 3))
        rotation_matrixes = np.zeros((nplanes, 3, 3))

        for index in range(3):
            xy_translation[:, index] = geo_pars['translation_%d' % index]
        for i in range(3):
            for j in range(3):
                rotation_matrixes[:, i, j] = geo_pars['rotation_%d_%d' % (i, j)]

        '''print "Translations:"
        print xy_translation
        print "Rotations:"
        print rotation_matrixes'''

        return xy_translation, rotation_matrixes


def update_geometry(geoFile, dut, xy_translation, rotation_matrixes, nplanes):
    '''Isn't it possible to just modify the value inside the h5 file instead of re-writing it?'''
    with tb.open_file(geoFile, mode='w') as geoFile:
        description = [('plane_number', np.int)]
        for index in range(3):
            description.append(('translation_%d' % index, np.float))
        for i in range(3):
            for j in range(3):
                description.append(('rotation_%d_%d' % (i, j), np.float))
        '''description.append(('sin_alpha', np.float))
        description.append(('sin_beta', np.float))
        description.append(('sin_gamma', np.float))'''
        ''' Is it necessary to specify the angles as well?'''

        geo_pars = np.zeros((nplanes,), dtype=description)
        for index in range(3):
            geo_pars['translation_%d' % index] = xy_translation[:, index]
        for i in range(3):
            for j in range(3):
                geo_pars['rotation_%d_%d' % (i, j)] = [rotation_matrixes[n][i, j] for n in range(nplanes)]

        geo_table = geoFile.create_table(geoFile.root, name='Geometry', title='File containing all the geometry parameters', description=np.zeros((1,), dtype=geo_pars.dtype).dtype, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        geo_table.append(geo_pars)


def update_rotation_val(geoFile, dut, i, j, val):
    xy_translation, rotation_matrixes = recontruct_geometry_from_file(geoFile)
    nplanes = xy_translation.shape[0]
    rotation_matrixes[dut, i, j] = val
    update_geometry(geoFile, dut, xy_translation, rotation_matrixes, nplanes)


def update_rotation_angle(geoFile, dut, val, mode="Absolute", angle="Gamma"):
    '''mode:
            --Absolute: move x and y to that specified values
            --Relative: increment the acutal x,y values by the specified ones
       angle:
            --Alpha: about x(column) axis
            --Beta:  about y(row) axis
            --Gamma: about z(beam) axis
        note here we use the rotation matrix as R_tot = R(alpha)R(beta)R(gamma), i.e.
        R_tot = [cosgamma*cosbeta,                                singamma*cosbeta,                                    -sinbeta]
                [(-singamma*cosalfa + sinalfa*sinbeta*cosgamma), (cosgamma*cosalfa +sinalfa*sinbeta*singamma), cosbeta*sinalpha]
                [...]'''

    if angle is not "Alpha" and angle is not "Beta" and angle is not "Gamma":
        print("update_rotation_angle: angle ", angle, " not recognized. Not applying any rotation.")
        return
    xy_translation, rotation_matrixes = recontruct_geometry_from_file(geoFile)
    nplanes = xy_translation.shape[0]
    tangamma = rotation_matrixes[dut, 0, 1] / rotation_matrixes[dut, 0, 0]

    # Extract the angles from the rotation matrix
    angles = {}
    angles["Gamma"] = asin(sqrt(tangamma ** 2 / (1 + tangamma ** 2)))
    angles["Beta"] = asin(-rotation_matrixes[dut, 0, 2])
    angles["Alpha"] = asin(rotation_matrixes[dut, 1, 2] / sqrt(1 - angles["Beta"] ** 2))

    if mode == "Absolute":
        angles[angle] = val
    elif mode == "Relative":
        angles[angle] += val

    singamma = sin(angles["Gamma"])
    sinbeta = sin(angles["Beta"])
    sinalpha = sin(angles["Alpha"])

    cosgamma = sqrt(1 - singamma ** 2)
    cosbeta = sqrt(1 - sinbeta ** 2)
    cosalpha = sqrt(1 - sinalpha ** 2)

    ''' Boring application: is there any method to produce rotation matrixes? '''
    rotation_matrixes[dut, 0, 0] = cosbeta * cosgamma
    rotation_matrixes[dut, 0, 1] = cosbeta * singamma
    rotation_matrixes[dut, 0, 2] = -sinbeta
    rotation_matrixes[dut, 1, 0] = (-singamma * cosalpha + sinalpha * sinbeta * cosgamma)
    rotation_matrixes[dut, 1, 1] = (cosgamma * cosalpha + sinalpha * sinbeta * singamma)
    rotation_matrixes[dut, 1, 2] = cosbeta * sinalpha
    rotation_matrixes[dut, 2, 0] = cosalpha * sinbeta * cosgamma + sinalpha * singamma
    rotation_matrixes[dut, 2, 1] = cosalpha * sinbeta * singamma - sinalpha * cosgamma
    rotation_matrixes[dut, 2, 2] = cosbeta * cosalpha

    update_geometry(geoFile, dut, xy_translation, rotation_matrixes, nplanes)


def update_rotation_angles(geoFile, dut, vals, mode="Absolute"):
    '''mode:
            --Absolute: move x and y to that specified values
            --Relative: increment the acutal x,y values by the specified ones
        note here we use the rotation matrix as R_tot = R(alpha)R(beta)R(gamma), i.e.
        R_tot = [cosgamma*cosbeta,                                singamma*cosbeta,                                    -sinbeta]
                [(-singamma*cosalfa + sinalfa*sinbeta*cosgamma), (cosgamma*cosalfa +sinalfa*sinbeta*singamma), cosbeta*sinalpha]
                [...]'''

    xy_translation, rotation_matrixes = recontruct_geometry_from_file(geoFile)
    nplanes = xy_translation.shape[0]
    tangamma = rotation_matrixes[dut, 0, 1] / rotation_matrixes[dut, 0, 0]
    angles = np.zeros(3)
    angles[2] = np.sign(tangamma) * asin(sqrt(tangamma ** 2 / (1 + tangamma ** 2)))
    angles[1] = asin(-rotation_matrixes[dut, 0, 2])
    angles[0] = asin(rotation_matrixes[dut, 1, 2] / sqrt(1 - angles[1] ** 2))

    if mode == "Absolute":
        angles = vals
    elif mode == "Relative":
        angles += vals

    singamma = sin(angles[2])
    sinbeta = sin(angles[1])
    sinalpha = sin(angles[0])

    cosgamma = sqrt(1 - singamma ** 2)
    cosbeta = sqrt(1 - sinbeta ** 2)
    cosalpha = sqrt(1 - sinalpha ** 2)
    ''' Boring application: is there any method to produce rotation matrixes? '''
    rotation_matrixes[dut, 0, 0] = cosbeta * cosgamma
    rotation_matrixes[dut, 0, 1] = cosbeta * singamma
    rotation_matrixes[dut, 0, 2] = -sinbeta
    rotation_matrixes[dut, 1, 0] = (-singamma * cosalpha + sinalpha * sinbeta * cosgamma)
    rotation_matrixes[dut, 1, 1] = (cosgamma * cosalpha + sinalpha * sinbeta * singamma)
    rotation_matrixes[dut, 1, 2] = cosbeta * sinalpha
    rotation_matrixes[dut, 2, 0] = cosalpha * sinbeta * cosgamma + sinalpha * singamma
    rotation_matrixes[dut, 2, 1] = cosalpha * sinbeta * singamma - sinalpha * cosgamma
    rotation_matrixes[dut, 2, 2] = cosbeta * cosalpha

    update_geometry(geoFile, dut, xy_translation, rotation_matrixes, nplanes)


def update_translation_val(geoFile, dut, xval, yval, mode="Absolute"):
    '''mode:
            --Absolute: move x and y to that specified values
            --Relative: increment the acutal x,y values by the specified ones'''

    xy_translation, rotation_matrixes = recontruct_geometry_from_file(geoFile)
    nplanes = xy_translation.shape[0]
    if mode == "Absolute":
        xy_translation[dut, 0] = xval
        xy_translation[dut, 1] = yval
    elif mode == "Relative":
        xy_translation[dut, 0] += xval
        xy_translation[dut, 1] += yval
    else:
        print("update_translation_val: mode not recognized. Not applying any translation.")
    update_geometry(geoFile, dut, xy_translation, rotation_matrixes, nplanes)


def modifiy_alignment(alignment_file):
    with tb.open_file(alignment_file, mode='r') as alignment:
        corrs = alignment.root.Alignment[:]

        for i in range(corrs.shape[0]):
            if corrs["c1"][i] > 0:
                corrs["c1"][i] = 1
            else:
                corrs["c1"][i] = -1
            corrs["c1_error"][i] = 50000
            corrs["sigma"][i] = 500

    with tb.open_file(alignment_file, mode='w') as alignment:
        try:
            result_table = alignment.create_table(alignment.root, name='Alignment', description=corrs.dtype, title='Correlation data', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            result_table.append(corrs)
        except tb.exceptions.NodeError:
            logging.warning('Correlation table exists already. Do not create new.')


def reset_geometry(geometry, dut):
    for index in range(2):
        geometry['translation_%d' % index][dut] = 0.
    for i in range(3):
        for j in range(3):
            if i == j:
                geometry['rotation_%d_%d' % (i, j)][dut] = 1.
            else:
                geometry['rotation_%d_%d' % (i, j)][dut] = 0.
