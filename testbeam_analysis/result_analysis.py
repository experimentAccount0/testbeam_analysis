''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''
from __future__ import division

import logging
import re
from collections import Iterable

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d
import os

from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import analysis_utils


def calculate_residuals(input_tracks_file, input_alignment_file, output_residuals_file, n_pixels, pixel_size, dut_names=None, use_duts=None, max_chi2=None, nbins_per_pixel=None, npixels_per_bin=None, force_prealignment=False, output_pdf=True, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    input_tracks_file : string
        File name with the tracks table
    input_alignment_file : pytables file
        File name of the input aligment data
    output_residuals_file : pytables file
        File name of the output file with the residual data
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension in um in column, row direction
        e.g. for 2 DUTs: pixel_size = [(250, 50), (250, 50)]
    use_duts : iterable
        The duts to calculate residuals for. If None all duts in the input_tracks_file are used
    max_chi2 : int, iterable, None
        Use only not heavily scattered tracks to increase track pointing resolution (cut on chi2).
        Cut can be a number and is used then for all DUTS or a list with a chi 2 cut for each DUT.
        If None no cut is applied.
    nbins_per_pixel : int
        Number of bins per pixel along the residual axis. Number is a positive integer or None to automatically set the binning.
    npixels_per_bin : int
        Number of pixels per bin along the position axis. Number is a positive integer or None to automatically set the binning.
    force_prealignment : boolean
        Take the prealignment, although if a coarse alignment is availale
    output_pdf : boolean. PDF pages object
        Set to True to create pdf plots with a file name output_residuals_file.pdf
        Set to None to show plots.
        Set to False to not create plots, saves a lot of time.
    chunk_size : integer
        The size of data in RAM
    Returns
    -------
    A list of residuals in column row. e.g.: [Col residual DUT 0, Row residual DUT 0, Col residual DUT 1, Row residual DUT 1, ...]
    '''
    logging.info('=== Calculate residuals ===')

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        if not use_prealignment:
            try:
                alignment = in_file_h5.root.Alignment[:]
            except tb.exceptions.NodeError:
                use_prealignment = True

    if use_prealignment:
        logging.info('Use prealignment data')
    else:
        logging.info('Use alignment data')

    close_pdf = True
    if output_pdf is True:
        output_fig = PdfPages(output_residuals_file[:-3] + '.pdf')
    elif output_pdf is None:
        output_fig = None
    elif output_pdf is False:
        output_fig = False
    else:
        output_fig = output_pdf  # output_fig is PDFpages object
        close_pdf = False  # Externally handled object do not close it

    residuals = []

    if not isinstance(max_chi2, Iterable):
        max_chi2 = [max_chi2] * n_duts

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_residuals_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                actual_max_chi2 = max_chi2[actual_dut]
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.debug('Calculate residuals for DUT %d', actual_dut)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    if actual_max_chi2:
                        tracks_chunk = tracks_chunk[tracks_chunk['track_chi2'] <= actual_max_chi2]
                    tracks_chunk = tracks_chunk[np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['y_dut_%d' % actual_dut]))]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    # Transform to local coordinate system
                    if use_prealignment:
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               prealignment=prealignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:  # Apply transformation from fine alignment information
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               alignment=alignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0) or not np.allclose(intersection_z_local, 0):
                        logging.error('Hit z position = %s and z intersection %s', str(hit_z_local[:3]), str(intersection_z_local[:3]))
                        raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

                    difference = np.column_stack((hit_x, hit_y, hit_z)) - np.column_stack((intersection_x, intersection_y, intersection_z))
                    difference_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local)) - np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))

                    # Histogram residuals in different ways
                    if initialize:  # Only true for the first iteration, calculate the binning for the histograms
                        initialize = False

                        # detect peaks and calculate width to estimate the size of the histograms
                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference[:, 0]), np.max(difference[:, 0])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][0] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][0] / nbins_per_pixel), pixel_size[actual_dut][0] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference[:, 0], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_x, fwhm_x, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            _, center_x, fwhm_x, _ = analysis_utils.simple_peak_detect(edge_center, hist)

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference[:, 1]), np.max(difference[:, 1])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][1] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][1] / nbins_per_pixel), pixel_size[actual_dut][1] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference[:, 1], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_y, fwhm_y, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            _, center_y, fwhm_y, _ = analysis_utils.simple_peak_detect(edge_center, hist)

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference_local[:, 0]), np.max(difference_local[:, 0])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][0] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][0] / nbins_per_pixel), pixel_size[actual_dut][0] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference_local[:, 0], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_col, fwhm_col, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            _, center_col, fwhm_col, _ = analysis_utils.simple_peak_detect(edge_center, hist)

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference_local[:, 1]), np.max(difference_local[:, 1])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][1] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][1] / nbins_per_pixel), pixel_size[actual_dut][1] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference_local[:, 1], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_row, fwhm_row, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            _, center_row, fwhm_row, _ = analysis_utils.simple_peak_detect(edge_center, hist)

                        # calculate the binning of the histograms, the minimum size is given by plot_npixels, otherwise FWHM is taken into account
                        plot_npixels = 6.0
                        if nbins_per_pixel is not None:
                            width = max(plot_npixels * pixel_size[actual_dut][0], pixel_size[actual_dut][0] * np.ceil(plot_npixels * fwhm_x / pixel_size[actual_dut][0]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][0]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][0])
                            x_range = (center_x - 0.5 * width, center_x + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][0] * np.ceil(plot_npixels * fwhm_x / pixel_size[actual_dut][0])
                            x_range = (center_x - width, center_x + width)
                        hist_residual_x_hist, hist_residual_x_xedges = np.histogram(difference[:, 0], range=x_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x), np.max(intersection_x)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][1], npixels_per_bin * pixel_size[actual_dut][1])
                        else:
                            nbins = "auto"
                        _, hist_residual_x_yedges = np.histogram(intersection_x, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_npixels * pixel_size[actual_dut][1], pixel_size[actual_dut][1] * np.ceil(plot_npixels * fwhm_x / pixel_size[actual_dut][1]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][1]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][1])
                            y_range = (center_y - 0.5 * width, center_y + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][1] * np.ceil(plot_npixels * fwhm_y / pixel_size[actual_dut][1])
                            y_range = (center_y - width, center_y + width)
                        hist_residual_y_hist, hist_residual_y_yedges = np.histogram(difference[:, 1], range=y_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y), np.max(intersection_y)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][0], npixels_per_bin * pixel_size[actual_dut][0])
                        else:
                            nbins = "auto"
                        _, hist_residual_y_xedges = np.histogram(intersection_y, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_npixels * pixel_size[actual_dut][0], pixel_size[actual_dut][0] * np.ceil(plot_npixels * fwhm_x / pixel_size[actual_dut][0]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][0]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][0])
                            col_range = (center_col - 0.5 * width, center_col + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][0] * np.ceil(plot_npixels * fwhm_col / pixel_size[actual_dut][0])
                            col_range = (center_col - width, center_col + width)
                        hist_residual_col_hist, hist_residual_col_xedges = np.histogram(difference_local[:, 0], range=col_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x_local), np.max(intersection_x_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][1], npixels_per_bin * pixel_size[actual_dut][1])
                        else:
                            nbins = "auto"
                        _, hist_residual_col_yedges = np.histogram(intersection_x_local, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_npixels * pixel_size[actual_dut][1], pixel_size[actual_dut][1] * np.ceil(plot_npixels * fwhm_x / pixel_size[actual_dut][1]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][1]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][1])
                            row_range = (center_row - 0.5 * width, center_row + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][1] * np.ceil(plot_npixels * fwhm_row / pixel_size[actual_dut][1])
                            row_range = (center_row - width, center_row + width)
                        hist_residual_row_hist, hist_residual_row_yedges = np.histogram(difference_local[:, 1], range=row_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y_local), np.max(intersection_y_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][0], npixels_per_bin * pixel_size[actual_dut][0])
                        else:
                            nbins = "auto"
                        _, hist_residual_row_xedges = np.histogram(intersection_y_local, bins=nbins)

                        # global x residual against x position
                        hist_x_residual_x_hist, hist_x_residual_x_xedges, hist_x_residual_x_yedges = np.histogram2d(
                            intersection_x,
                            difference[:, 0],
                            bins=(hist_residual_x_yedges, hist_residual_x_xedges))

                        # global y residual against y position
                        hist_y_residual_y_hist, hist_y_residual_y_xedges, hist_y_residual_y_yedges = np.histogram2d(
                            intersection_y,
                            difference[:, 1],
                            bins=(hist_residual_y_xedges, hist_residual_y_yedges))

                        # global y residual against x position
                        hist_x_residual_y_hist, hist_x_residual_y_xedges, hist_x_residual_y_yedges = np.histogram2d(
                            intersection_x,
                            difference[:, 1],
                            bins=(hist_residual_x_yedges, hist_residual_y_yedges))

                        # global x residual against y position
                        hist_y_residual_x_hist, hist_y_residual_x_xedges, hist_y_residual_x_yedges = np.histogram2d(
                            intersection_y,
                            difference[:, 0],
                            bins=(hist_residual_y_xedges, hist_residual_x_xedges))

                        # local column residual against column position
                        hist_col_residual_col_hist, hist_col_residual_col_xedges, hist_col_residual_col_yedges = np.histogram2d(
                            intersection_x_local,
                            difference_local[:, 0],
                            bins=(hist_residual_col_yedges, hist_residual_col_xedges))

                        # local row residual against row position
                        hist_row_residual_row_hist, hist_row_residual_row_xedges, hist_row_residual_row_yedges = np.histogram2d(
                            intersection_y_local,
                            difference_local[:, 1],
                            bins=(hist_residual_row_xedges, hist_residual_row_yedges))

                        # local row residual against column position
                        hist_col_residual_row_hist, hist_col_residual_row_xedges, hist_col_residual_row_yedges = np.histogram2d(
                            intersection_x_local,
                            difference_local[:, 1],
                            bins=(hist_residual_col_yedges, hist_residual_row_yedges))

                        # local column residual against row position
                        hist_row_residual_col_hist, hist_row_residual_col_xedges, hist_row_residual_col_yedges = np.histogram2d(
                            intersection_y_local,
                            difference_local[:, 0],
                            bins=(hist_residual_row_xedges, hist_residual_col_xedges))

                    else:  # adding data to existing histograms
                        hist_residual_x_hist += np.histogram(difference[:, 0], bins=hist_residual_x_xedges)[0]
                        hist_residual_y_hist += np.histogram(difference[:, 1], bins=hist_residual_y_yedges)[0]
                        hist_residual_col_hist += np.histogram(difference_local[:, 0], bins=hist_residual_col_xedges)[0]
                        hist_residual_row_hist += np.histogram(difference_local[:, 1], bins=hist_residual_row_yedges)[0]

                        # global x residual against x position
                        hist_x_residual_x_hist += np.histogram2d(
                            intersection_x,
                            difference[:, 0],
                            bins=(hist_x_residual_x_xedges, hist_x_residual_x_yedges))[0]

                        # global y residual against y position
                        hist_y_residual_y_hist += np.histogram2d(
                            intersection_y,
                            difference[:, 1],
                            bins=(hist_y_residual_y_xedges, hist_y_residual_y_yedges))[0]

                        # global y residual against x position
                        hist_x_residual_y_hist += np.histogram2d(
                            intersection_x,
                            difference[:, 1],
                            bins=(hist_x_residual_y_xedges, hist_x_residual_y_yedges))[0]

                        # global x residual against y position
                        hist_y_residual_x_hist += np.histogram2d(
                            intersection_y,
                            difference[:, 0],
                            bins=(hist_y_residual_x_xedges, hist_y_residual_x_yedges))[0]

                        # local column residual against column position
                        hist_col_residual_col_hist += np.histogram2d(
                            intersection_x_local,
                            difference_local[:, 0],
                            bins=(hist_col_residual_col_xedges, hist_col_residual_col_yedges))[0]

                        # local row residual against row position
                        hist_row_residual_row_hist += np.histogram2d(
                            intersection_y_local,
                            difference_local[:, 1],
                            bins=(hist_row_residual_row_xedges, hist_row_residual_row_yedges))[0]

                        # local row residual against column position
                        hist_col_residual_row_hist += np.histogram2d(
                            intersection_x_local,
                            difference_local[:, 1],
                            bins=(hist_col_residual_row_xedges, hist_col_residual_row_yedges))[0]

                        # local column residual against row position
                        hist_row_residual_col_hist += np.histogram2d(
                            intersection_y_local,
                            difference_local[:, 0],
                            bins=(hist_row_residual_col_xedges, hist_row_residual_col_yedges))[0]

                logging.debug('Storing residual histograms...')

                dut_name = dut_names[actual_dut] if dut_names else ("DUT " + str(actual_dut))
                # Global residuals
                fit_residual_x, cov_residual_x = analysis_utils.fit_residuals(
                    hist=hist_residual_x_hist,
                    edges=hist_residual_x_xedges,
                    label='X residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_res_x = out_file_h5.create_carray(out_file_h5.root,
                                                      name='ResidualsX_DUT%d' % (actual_dut),
                                                      title='Residual distribution in x direction for %s' % (dut_name),
                                                      atom=tb.Atom.from_dtype(hist_residual_x_hist.dtype),
                                                      shape=hist_residual_x_hist.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_x.attrs.xedges = hist_residual_x_xedges
                out_res_x.attrs.fit_coeff = fit_residual_x
                out_res_x.attrs.fit_cov = cov_residual_x
                out_res_x[:] = hist_residual_x_hist

                fit_residual_y, cov_residual_y = analysis_utils.fit_residuals(
                    hist=hist_residual_y_hist,
                    edges=hist_residual_y_yedges,
                    label='Y residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_res_y = out_file_h5.create_carray(out_file_h5.root,
                                                      name='ResidualsY_DUT%d' % (actual_dut),
                                                      title='Residual distribution in y direction for %s' % (dut_name),
                                                      atom=tb.Atom.from_dtype(hist_residual_y_hist.dtype),
                                                      shape=hist_residual_y_hist.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_y.attrs.yedges = hist_residual_y_yedges
                out_res_y.attrs.fit_coeff = fit_residual_y
                out_res_y.attrs.fit_cov = cov_residual_y
                out_res_y[:] = hist_residual_y_hist

                fit_x_residual_x, cov_x_residual_x = analysis_utils.fit_residuals_vs_position(
                    hist=hist_x_residual_x_hist,
                    xedges=hist_x_residual_x_xedges,
                    yedges=hist_x_residual_x_yedges,
                    xlabel='X position [um]',
                    ylabel='X residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )

                out_x_res_x = out_file_h5.create_carray(out_file_h5.root,
                                                        name='XResidualsX_DUT%d' % (actual_dut),
                                                        title='Residual distribution in x direction as a function of the x position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_x_residual_x_hist.dtype),
                                                        shape=hist_x_residual_x_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_x.attrs.xedges = hist_x_residual_x_xedges
                out_x_res_x.attrs.yedges = hist_x_residual_x_yedges
                out_x_res_x.attrs.fit_coeff = fit_x_residual_x
                out_x_res_x.attrs.fit_cov = cov_x_residual_x
                out_x_res_x[:] = hist_x_residual_x_hist

                fit_y_residual_y, cov_y_residual_y = analysis_utils.fit_residuals_vs_position(
                    hist=hist_y_residual_y_hist,
                    xedges=hist_y_residual_y_xedges,
                    yedges=hist_y_residual_y_yedges,
                    xlabel='Y position [um]',
                    ylabel='Y residual [um]',
                    title='Residuals for DUT %d' % actual_dut,
                    output_fig=output_fig
                )
                out_y_res_y = out_file_h5.create_carray(out_file_h5.root,
                                                        name='YResidualsY_DUT%d' % (actual_dut),
                                                        title='Residual distribution in y direction as a function of the y position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_y_residual_y_hist.dtype),
                                                        shape=hist_y_residual_y_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_y.attrs.xedges = hist_y_residual_y_xedges
                out_y_res_y.attrs.yedges = hist_y_residual_y_yedges
                out_y_res_y.attrs.fit_coeff = fit_y_residual_y
                out_y_res_y.attrs.fit_cov = cov_y_residual_y
                out_y_res_y[:] = hist_y_residual_y_hist

                fit_x_residual_y, cov_x_residual_y = analysis_utils.fit_residuals_vs_position(
                    hist=hist_x_residual_y_hist,
                    xedges=hist_x_residual_y_xedges,
                    yedges=hist_x_residual_y_yedges,
                    xlabel='X position [um]',
                    ylabel='Y residual [um]',
                    title='Residuals for DUT %d' % actual_dut,
                    output_fig=output_fig
                )
                out_x_res_y = out_file_h5.create_carray(out_file_h5.root,
                                                        name='XResidualsY_DUT%d' % (actual_dut),
                                                        title='Residual distribution in y direction as a function of the x position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_x_residual_y_hist.dtype),
                                                        shape=hist_x_residual_y_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_y.attrs.xedges = hist_x_residual_y_xedges
                out_x_res_y.attrs.yedges = hist_x_residual_y_yedges
                out_x_res_y.attrs.fit_coeff = fit_x_residual_y
                out_x_res_y.attrs.fit_cov = cov_x_residual_y
                out_x_res_y[:] = hist_x_residual_y_hist

                fit_y_residual_x, cov_y_residual_x = analysis_utils.fit_residuals_vs_position(
                    hist=hist_y_residual_x_hist,
                    xedges=hist_y_residual_x_xedges,
                    yedges=hist_y_residual_x_yedges,
                    xlabel='Y position [um]',
                    ylabel='X residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_y_res_x = out_file_h5.create_carray(out_file_h5.root,
                                                        name='YResidualsX_DUT%d' % (actual_dut),
                                                        title='Residual distribution in x direction as a function of the y position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_y_residual_x_hist.dtype),
                                                        shape=hist_y_residual_x_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_x.attrs.xedges = hist_y_residual_x_xedges
                out_y_res_x.attrs.yedges = hist_y_residual_x_yedges
                out_y_res_x.attrs.fit_coeff = fit_y_residual_x
                out_y_res_x.attrs.fit_cov = cov_y_residual_x
                out_y_res_x[:] = hist_y_residual_x_hist

                # Local residuals
                fit_residual_col, cov_residual_col = analysis_utils.fit_residuals(
                    hist=hist_residual_col_hist,
                    edges=hist_residual_col_xedges,
                    label='Column residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_res_col = out_file_h5.create_carray(out_file_h5.root,
                                                        name='ResidualsCol_DUT%d' % (actual_dut),
                                                        title='Residual distribution in column direction for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_residual_col_hist.dtype),
                                                        shape=hist_residual_col_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_col.attrs.xedges = hist_residual_col_xedges
                out_res_col.attrs.fit_coeff = fit_residual_col
                out_res_col.attrs.fit_cov = cov_residual_col
                out_res_col[:] = hist_residual_col_hist

                fit_residual_row, cov_residual_row = analysis_utils.fit_residuals(
                    hist=hist_residual_row_hist,
                    edges=hist_residual_row_yedges,
                    label='Row residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_res_row = out_file_h5.create_carray(out_file_h5.root,
                                                        name='ResidualsRow_DUT%d' % (actual_dut),
                                                        title='Residual distribution in row direction for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_residual_row_hist.dtype),
                                                        shape=hist_residual_row_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_row.attrs.yedges = hist_residual_row_yedges
                out_res_row.attrs.fit_coeff = fit_residual_row
                out_res_row.attrs.fit_cov = cov_residual_row
                out_res_row[:] = hist_residual_row_hist

                fit_col_residual_col, cov_col_residual_col = analysis_utils.fit_residuals_vs_position(
                    hist=hist_col_residual_col_hist,
                    xedges=hist_col_residual_col_xedges,
                    yedges=hist_col_residual_col_yedges,
                    xlabel='Column position [um]',
                    ylabel='Column residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_col_res_col = out_file_h5.create_carray(out_file_h5.root,
                                                            name='ColResidualsCol_DUT%d' % (actual_dut),
                                                            title='Residual distribution in column direction as a function of the column position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_col_residual_col_hist.dtype),
                                                            shape=hist_col_residual_col_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_col.attrs.xedges = hist_col_residual_col_xedges
                out_col_res_col.attrs.yedges = hist_col_residual_col_yedges
                out_col_res_col.attrs.fit_coeff = fit_col_residual_col
                out_col_res_col.attrs.fit_cov = cov_col_residual_col
                out_col_res_col[:] = hist_col_residual_col_hist

                fit_row_residual_row, cov_row_residual_row = analysis_utils.fit_residuals_vs_position(
                    hist=hist_row_residual_row_hist,
                    xedges=hist_row_residual_row_xedges,
                    yedges=hist_row_residual_row_yedges,
                    xlabel='Row position [um]',
                    ylabel='Row residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_row_res_row = out_file_h5.create_carray(out_file_h5.root,
                                                            name='RowResidualsRow_DUT%d' % (actual_dut),
                                                            title='Residual distribution in row direction as a function of the row position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_row_residual_row_hist.dtype),
                                                            shape=hist_row_residual_row_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_row.attrs.xedges = hist_row_residual_row_xedges
                out_row_res_row.attrs.yedges = hist_row_residual_row_yedges
                out_row_res_row.attrs.fit_coeff = fit_row_residual_row
                out_row_res_row.attrs.fit_cov = cov_row_residual_row
                out_row_res_row[:] = hist_row_residual_row_hist

                fit_col_residual_row, cov_col_residual_row = analysis_utils.fit_residuals_vs_position(
                    hist=hist_col_residual_row_hist,
                    xedges=hist_col_residual_row_xedges,
                    yedges=hist_col_residual_row_yedges,
                    xlabel='Column position [um]',
                    ylabel='Row residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_col_res_row = out_file_h5.create_carray(out_file_h5.root,
                                                            name='ColResidualsRow_DUT%d' % (actual_dut),
                                                            title='Residual distribution in row direction as a function of the column position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_col_residual_row_hist.dtype),
                                                            shape=hist_col_residual_row_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_row.attrs.xedges = hist_col_residual_row_xedges
                out_col_res_row.attrs.yedges = hist_col_residual_row_yedges
                out_col_res_row.attrs.fit_coeff = fit_col_residual_row
                out_col_res_row.attrs.fit_cov = cov_col_residual_row
                out_col_res_row[:] = hist_col_residual_row_hist

                fit_row_residual_col, cov_row_residual_col = analysis_utils.fit_residuals_vs_position(
                    hist=hist_row_residual_col_hist,
                    xedges=hist_row_residual_col_xedges,
                    yedges=hist_row_residual_col_yedges,
                    xlabel='Row position [um]',
                    ylabel='Column residual [um]',
                    title='Residuals for %s' % (dut_name),
                    output_fig=output_fig
                )
                out_row_res_col = out_file_h5.create_carray(out_file_h5.root,
                                                            name='RowResidualsCol_DUT%d' % (actual_dut),
                                                            title='Residual distribution in column direction as a function of the row position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_row_residual_col_hist.dtype),
                                                            shape=hist_row_residual_col_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_col.attrs.xedges = hist_row_residual_col_xedges
                out_row_res_col.attrs.yedges = hist_row_residual_col_yedges
                out_row_res_col.attrs.fit_coeff = fit_row_residual_col
                out_row_res_col.attrs.fit_cov = cov_row_residual_col
                out_row_res_col[:] = hist_row_residual_col_hist

    if output_fig and close_pdf:
        output_fig.close()


def calculate_efficiency(input_tracks_file, input_alignment_file, output_pdf, bin_size, sensor_size, pixel_size=None, n_pixels=None, minimum_track_density=1, max_distance=500, use_duts=None, max_chi2=None, force_prealignment=False, cut_distance=None, col_range=None, row_range=None, show_inefficient_events=False, output_file=None, chunk_size=1000000):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.
    Parameters
    ----------
    input_tracks_file : string
        file name with the tracks table
    input_alignment_file : pytables file
        File name of the input aligment data
    output_pdf : pdf file name object
    bin_size : iterable
        sizes of bins (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes
    sensor_size : Tuple or list of tuples
        Describes the sensor size for each DUT. If one tuple is given it is (size x, size y)
        If several tuples are given it is [(DUT0 size x, DUT0 size y), (DUT1 size x, DUT1 size y), ...]
    minimum_track_density : int
        minimum track density required to consider bin for efficiency calculation
    use_duts : iterable
        the DUTs to calculate efficiency for. If None all duts are used
    max_chi2 : int
        only use track with a chi2 <= max_chi2
    force_prealignment : boolean
        Take the prealignment, although if a coarse alignment is availale
    cut_distance : int
        use only distances (between DUT hit and track hit) smaller than cut_distance
    max_distance : int
        defines binnig of distance values
    col_range, row_range : iterable
        column / row value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation)
    chunk_size : integer
        The size of data in RAM
    '''
    logging.info('=== Calculate efficiency ===')

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        if not use_prealignment:
            try:
                alignment = in_file_h5.root.Alignment[:]
                logging.info('Use alignment data')
            except tb.exceptions.NodeError:
                use_prealignment = True
                logging.info('Use prealignment data')

    if not isinstance(max_chi2, Iterable):
        max_chi2 = [max_chi2] * n_duts

    with PdfPages(output_pdf) as output_fig:
        efficiencies = []
        pass_tracks = []
        total_tracks = []
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for index, node in enumerate(in_file_h5.root):
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Calculate efficiency for DUT %d', actual_dut)

                # Calculate histogram properties (bins size and number of bins)
                bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
                if len(bin_size) != 1:
                    actual_bin_size_x = bin_size[index][0]
                    actual_bin_size_y = bin_size[index][1]
                else:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]
                dimensions = [sensor_size, ] if not isinstance(sensor_size, Iterable) else sensor_size  # Sensor dimensions for each DUT
                if len(dimensions) == 1:
                    dimensions = dimensions[0]
                else:
                    dimensions = dimensions[index]
                n_bin_x = int(dimensions[0] / actual_bin_size_x)
                n_bin_y = int(dimensions[1] / actual_bin_size_y)

                # Define result histograms, these are filled for each hit chunk
#                 total_distance_array = np.zeros(shape=(n_bin_x, n_bin_y, max_distance))
                total_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)
                total_track_density = np.zeros(shape=(n_bin_x, n_bin_y))
                total_track_density_with_DUT_hit = np.zeros(shape=(n_bin_x, n_bin_y))

                actual_max_chi2 = max_chi2[index]

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Cut in Chi 2 of the track fit
                    if actual_max_chi2:
                        tracks_chunk = tracks_chunk[tracks_chunk['track_chi2'] <= max_chi2]

                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    # Transform to local coordinate system
                    if use_prealignment:
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               prealignment=prealignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:  # Apply transformation from alignment information
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               alignment=alignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    # Quickfix that center of sensor is local system is in the center and not at the edge
                    hit_x_local, hit_y_local = hit_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    intersection_x_local, intersection_y_local = intersection_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], intersection_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]

                    intersections_local = np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))
                    hits_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local))

                    if not np.allclose(hits_local[np.isfinite(hits_local[:, 2]), 2], 0.0) or not np.allclose(intersection_z_local, 0.0):
                        raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

                    # Usefull for debugging, print some inefficient events that can be cross checked
                    # Select virtual hits
                    sel_virtual = np.isnan(tracks_chunk['x_dut_%d' % actual_dut])
                    if show_inefficient_events:
                        logging.info('These events are inefficient: %s', str(tracks_chunk['event_number'][sel_virtual]))

                    # Select hits from column, row range (e.g. to supress edge pixels)
                    col_range = [col_range, ] if not isinstance(col_range, Iterable) else col_range
                    row_range = [row_range, ] if not isinstance(row_range, Iterable) else row_range
                    if len(col_range) == 1:
                        index = 0
                    if len(row_range) == 1:
                        index = 0
                    if col_range[index] is not None:
                        selection = np.logical_and(intersections_local[:, 0] >= col_range[index][0], intersections_local[:, 0] <= col_range[index][1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]
                    if row_range[index] is not None:
                        selection = np.logical_and(intersections_local[:, 1] >= row_range[index][0], intersections_local[:, 1] <= row_range[index][1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]

                    # Calculate distance between track hit and DUT hit
                    scale = np.square(np.array((1, 1, 0)))  # Regard pixel size for calculating distances
                    distance = np.sqrt(np.dot(np.square(intersections_local - hits_local), scale))  # Array with distances between DUT hit and track hit for each event. Values in um

                    col_row_distance = np.column_stack((hits_local[:, 0], hits_local[:, 1], distance))

#                     total_distance_array += np.histogramdd(col_row_distance, bins=(n_bin_x, n_bin_y, max_distance), range=[[0, dimensions[0]], [0, dimensions[1]], [0, max_distance]])[0]
                    total_hit_hist += (np.histogram2d(hits_local[:, 0], hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]).astype(np.uint32)
#                     total_hit_hist += (np.histogram2d(hits_local[:, 0], hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[-dimensions[0] / 2., dimensions[0] / 2.], [-dimensions[1] / 2., dimensions[1] / 2.]])[0]).astype(np.uint32)

                    # Calculate efficiency
                    selection = ~np.isnan(hits_local[:, 0])
                    if cut_distance:  # Select intersections where hit is in given distance around track intersection
                        intersection_valid_hit = intersections_local[np.logical_and(selection, distance < cut_distance)]
                    else:
                        intersection_valid_hit = intersections_local[selection]

                    total_track_density += np.histogram2d(intersections_local[:, 0], intersections_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]
                    total_track_density_with_DUT_hit += np.histogram2d(intersection_valid_hit[:, 0], intersection_valid_hit[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]

                    if np.all(total_track_density == 0):
                        logging.warning('No tracks on DUT %d, cannot calculate efficiency', actual_dut)
                        continue

                efficiency = np.zeros_like(total_track_density_with_DUT_hit)
                efficiency[total_track_density != 0] = total_track_density_with_DUT_hit[total_track_density != 0].astype(np.float) / total_track_density[total_track_density != 0].astype(np.float) * 100.

                efficiency = np.ma.array(efficiency, mask=total_track_density < minimum_track_density)

                if not np.any(efficiency):
                    raise RuntimeError('All efficiencies for DUT%d are zero, consider changing cut values!', actual_dut)

                # Calculate distances between hit and intersection
#                 distance_mean_array = np.average(total_distance_array, axis=2, weights=range(0, max_distance)) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_mean_array = np.ma.masked_invalid(distance_mean_array)
#                 distance_max_array = np.amax(total_distance_array, axis=2) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_min_array = np.amin(total_distance_array, axis=2) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_max_array = np.ma.masked_invalid(distance_max_array)
#                 distance_min_array = np.ma.masked_invalid(distance_min_array)

#                 plot_utils.plot_track_distances(distance_min_array, distance_max_array, distance_mean_array)
                plot_utils.efficiency_plots(total_hit_hist, total_track_density, total_track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, plot_range=dimensions, cut_distance=cut_distance, output_fig=output_fig)

                logging.info('Efficiency =  %1.4f +- %1.4f', np.ma.mean(efficiency), np.ma.std(efficiency))
                efficiencies.append(np.ma.mean(efficiency))

                if output_file:
                    with tb.open_file(output_file, 'a') as out_file_h5:
                        try:
                            actual_dut_folder = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)
                        except tb.NodeError:
                            logging.warning('Data for DUT%d exists already and will be overwritten', actual_dut)
                            out_file_h5.remove_node('/DUT_%d' % actual_dut, recursive=True)
                            actual_dut_folder = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)

                        out_efficiency = out_file_h5.create_carray(actual_dut_folder, name='Efficiency', title='Efficiency map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.dtype), shape=efficiency.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                        out_efficiency_mask = out_file_h5.create_carray(actual_dut_folder, name='Efficiency_mask', title='Masked pixel map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.mask.dtype), shape=efficiency.mask.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                        # For correct statistical error calculation the number of detected tracks over total tracks is needed
                        out_pass = out_file_h5.create_carray(actual_dut_folder, name='Passing_tracks', title='Passing events of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_track_density_with_DUT_hit.dtype), shape=total_track_density_with_DUT_hit.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                        out_total = out_file_h5.create_carray(actual_dut_folder, name='Total_tracks', title='Total events of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_track_density.dtype), shape=total_track_density.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                        pass_tracks.append(total_track_density_with_DUT_hit.sum())
                        total_tracks.append(total_track_density.sum())
                        logging.info('Passing / total tracks: %d / %d', total_track_density_with_DUT_hit.sum(), total_track_density.sum())

                        # Store parameters used for efficiency calculation
                        out_efficiency.attrs.bin_size = bin_size
                        out_efficiency.attrs.minimum_track_density = minimum_track_density
                        out_efficiency.attrs.sensor_size = sensor_size
                        out_efficiency.attrs.use_duts = use_duts
                        out_efficiency.attrs.max_chi2 = max_chi2
                        out_efficiency.attrs.cut_distance = cut_distance
                        out_efficiency.attrs.max_distance = max_distance
                        out_efficiency.attrs.col_range = col_range
                        out_efficiency.attrs.row_range = row_range
                        out_efficiency[:] = efficiency.T
                        out_efficiency_mask[:] = efficiency.mask.T
                        out_pass[:] = total_track_density_with_DUT_hit.T
                        out_total[:] = total_track_density.T
    return efficiencies, pass_tracks, total_tracks
