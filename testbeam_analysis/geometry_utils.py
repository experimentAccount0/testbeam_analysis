from __future__ import division

import os
import logging
from math import sqrt

import tables as tb
import numpy as np


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
            print "create_initial_geometry: number of planes in initial translation doesn't match the one given by zpos"
            print "create_initial_geometry: setting all translations to 0"

        if initial_rotation is not None:
            if initial_rotation.shape == rotation_matrixes.shape:
                rotation_matrixes = initial_rotation
            else:
                print "create_initial_geometry: initial_rotation must have shape (n_planes,3,3)"
                print "create_initial_geometry: setting all rotations to Identity"

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
        print "update_rotation_angle: angle ", angle, " not recognized. Not applying any rotation."
        return
    xy_translation, rotation_matrixes = recontruct_geometry_from_file(geoFile)
    nplanes = xy_translation.shape[0]
    tangamma = rotation_matrixes[dut, 0, 1] / rotation_matrixes[dut, 0, 0]
    angles = {}
    angles["Gamma"] = sqrt(tangamma ** 2 / (1 + tangamma ** 2))
    angles["Beta"] = -rotation_matrixes[dut, 0, 2]
    angles["Alpha"] = rotation_matrixes[dut, 1, 2] / sqrt(1 - angles["Beta"] ** 2)
    '''Not expecting big angles, operating in the approximation sin(theta) ~ theta '''
    if mode == "Absolute":
        angles[angle] = val
    elif mode == "Relative":
        angles[angle] += val

    cosgamma = sqrt(1 - angles["Gamma"] ** 2)
    cosbeta = sqrt(1 - angles["Beta"] ** 2)
    cosalpha = sqrt(1 - angles["Alpha"] ** 2)
    ''' Boring application: is there any method to produce rotation matrixes? '''
    rotation_matrixes[dut, 0, 0] = cosbeta * cosgamma
    rotation_matrixes[dut, 0, 1] = cosbeta * angles["Gamma"]
    rotation_matrixes[dut, 0, 2] = -angles["Beta"]
    rotation_matrixes[dut, 1, 0] = (-angles["Gamma"] * cosalpha + angles["Alpha"] * angles["Beta"] * cosgamma)
    rotation_matrixes[dut, 1, 1] = (cosgamma * cosalpha + angles["Alpha"] * angles["Beta"] * angles["Gamma"])
    rotation_matrixes[dut, 1, 2] = cosbeta * angles["Alpha"]
    rotation_matrixes[dut, 2, 0] = cosalpha * angles["Beta"] * cosgamma + angles["Alpha"] * angles["Gamma"]
    rotation_matrixes[dut, 2, 1] = cosalpha * angles["Beta"] * angles["Gamma"] - angles["Alpha"] * cosgamma
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
        print "update_translation_val: mode not recognized. Not applying any translation."
    update_geometry(geoFile, dut, xy_translation, rotation_matrixes, nplanes)
