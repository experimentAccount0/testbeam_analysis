from __future__ import division

import logging
from math import sqrt

import tables as tb
import numpy as np

from math import sin
from math import asin


def rotation_matrix_x(angle):
    ''' Calculates the rotation matrix for the rotation around the x axis by an angle
    in a cartesian left-handed coordinate system

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
                     [0, -np.sin(angle), np.cos(angle)]]).T


def rotation_matrix_y(angle):
    ''' Calculates the rotation matrix for the rotation around the y axis by an angle
    in a cartesian left-handed coordinate system

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
                     [np.sin(angle), 0, np.cos(angle)]]).T


def rotation_matrix_z(angle):
    ''' Calculates the rotation matrix for the rotation around the z axis by an angle
    in a cartesian left-handed coordinate system

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
                     [0, 0, 1]]).T


def rotation_matrix(alpha, beta, gamma):
    ''' Calculates the rotation matrix for the rotation around the three cartesian axis x, y, z
    in a left-handed system. The rotation is done around x then y then z.

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
    in a cartesian left-handed system.

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
    ''' Calculates the transformation matrix that applies a translation by x, y, z
    to the local coordinate system and then a rotation in the local coordinate system.

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

    # Extend rotation matrix R by one dimension
    R = np.eye(4, 4, 0)
    R[:3, :3] = rotation_matrix(alpha, beta, gamma)

    # Get translation matrix T
    T = translation_matrix(x, y, z)

    return np.dot(T, R)


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
    R[:3, :3] = rotation_matrix(alpha, beta, gamma).T  # Inverse of a rotation matrix is also the transformed matrix, since Det = 1

    # Get inverse translation matrix T
    T = translation_matrix(-x, -y, -z)

    return np.dot(R, T)


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
