''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''
from __future__ import division

import logging
import re
from collections import Iterable

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import analysis_utils


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def calculate_residuals(input_tracks_file, input_alignment_file, output_residuals_file, n_pixels, pixel_size, use_duts=None, max_chi2=None, force_prealignment=False, output_pdf=True, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.
    Parameters
    ----------
    input_tracks_file : string
        File name with the tracks table
    input_alignment_file : pytables file
        File name of the input aligment data
    output_residuals_file : pytables file
        File name of the output file with the residual data
    n_pixels
    pixel_size
    use_duts : iterable
        The duts to calculate residuals for. If None all duts in the input_tracks_file are used
    max_chi2 : int
        Use only converged fits (cut on chi2)
    force_prealignment : boolean
        Take the prealignment, although if a coarse alignment is availale
    output_pdf : boolean
        Set to true to create plots.
    Returns
    -------
    A list of residuals in column row. e.g.: [Col residual DUT 0, Row residual DUT 0, Col residual DUT 1, Row residual DUT 1, ...]
    '''
    logging.info('=== Calculate residuals ===')

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        if not use_prealignment:
            try:
                alignment = in_file_h5.root.Alignment[:]
            except tb.exceptions.NodeError:
                use_prealignment = True

    if use_prealignment:
        logging.info('Use prealignment data')
    else:
        logging.info('Use alignment data')

    output_fig = PdfPages(output_residuals_file[:-3] + '.pdf') if output_pdf is not False else False

    residuals = []

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_residuals_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.debug('Calculate residuals for DUT %d', actual_dut)

                mean_column, std_column = None, None  # column residual mean and RMS needed for histogramming range determination
                mean_row, std_row = None, None  # row residual mean and RMS needed for histogramming range determination

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    if max_chi2:
                        tracks_chunk = tracks_chunk[tracks_chunk['track_chi2'] <= max_chi2]
                    tracks_chunk = tracks_chunk[np.logical_and(tracks_chunk['x_dut_%d' % actual_dut] != 0., tracks_chunk['y_dut_%d' % actual_dut] != 0.)]  # take only tracks where actual dut has a hit, otherwise residual wrong

                    # Transform the hits and track intersections into the local coordinate system
                    if not use_prealignment:  # Use alignment with rotations
                        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=alignment[actual_dut]['translation_x'],
                                                                                                     y=alignment[actual_dut]['translation_y'],
                                                                                                     z=alignment[actual_dut]['translation_z'],
                                                                                                     alpha=alignment[actual_dut]['alpha'],
                                                                                                     beta=alignment[actual_dut]['beta'],
                                                                                                     gamma=alignment[actual_dut]['gamma'])
                    else:  # Prealignment only gives offsets and no rotations, thus apply the known offsets
                        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=prealignment[actual_dut]['column_c0'],
                                                                                                     y=prealignment[actual_dut]['row_c0'],
                                                                                                     z=prealignment[actual_dut]['z'],
                                                                                                     alpha=0,
                                                                                                     beta=0,
                                                                                                     gamma=0)

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    # Coordinates in local coordinate system (x, y, z) -> (col, row, 0)
                    hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_transformation_matrix(x=hit_x,
                                                                                                       y=hit_y,
                                                                                                       z=hit_z,
                                                                                                       transformation_matrix=transformation_matrix)

                    intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_transformation_matrix(x=intersection_x,
                                                                                                                                  y=intersection_y,
                                                                                                                                  z=intersection_z,
                                                                                                                                  transformation_matrix=transformation_matrix)

                    if not np.allclose(hit_z_local, 0) or not np.allclose(intersection_z_local, 0):
                        logging.error('Hit z position = %s and z intersection %s', str(hit_z_local[:3]), str(intersection_z_local[:3]))
                        raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

                    difference = np.column_stack((hit_x, hit_y, hit_z)) - np.column_stack((intersection_x, intersection_y, intersection_z))
                    difference_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local)) - np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))

                    # Histogram residuals in different ways
                    if not mean_column:
                        mean_x, std_x = np.mean(difference[:, 0]), np.std(difference[:, 0])
                        mean_y, std_y = np.mean(difference[:, 1]), np.std(difference[:, 1])
                        mean_column, std_column = np.mean(difference_local[:, 0]), np.std(difference_local[:, 0])
                        mean_row, std_row = np.mean(difference_local[:, 1]), np.std(difference_local[:, 1])

                        hist_residual_x = np.histogram(difference[:, 0], range=(mean_x - 5. * std_x, mean_x + 5. * std_x), bins=1000)
                        hist_residual_y = np.histogram(difference[:, 1], range=(mean_y - 5. * std_y, mean_y + 5. * std_y), bins=1000)
                        hist_residual_col = np.histogram(difference_local[:, 0], range=(mean_column - 5. * std_column, mean_column + 5. * std_column), bins=1000)
                        hist_residual_row = np.histogram(difference_local[:, 1], range=(mean_row - 5. * std_row, mean_row + 5. * std_row), bins=1000)

                        # X residual agains x position
                        hist_x_residual_x = np.histogram2d(intersection_x,
                                                           difference[:, 0],
                                                           bins=(200, 800),
                                                           range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_x - 1 * std_x, mean_x + 1 * std_x)))

                        # Y residual agains y position
                        hist_y_residual_y = np.histogram2d(intersection_y,
                                                           difference[:, 1],
                                                           bins=(200, 800),
                                                           range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_y - 1 * std_y, mean_y + 1 * std_y)))

                        # Y residual agains x position
                        hist_x_residual_y = np.histogram2d(intersection_x,
                                                           difference[:, 1],
                                                           bins=(200, 800),
                                                           range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_y - 1 * std_y, mean_y + 1 * std_y)))

                        # X residual agains y position
                        hist_y_residual_x = np.histogram2d(intersection_y,
                                                           difference[:, 0],
                                                           bins=(200, 800),
                                                           range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_x - 1 * std_x, mean_x + 1 * std_x)))

                        # Column residual agains column position
                        hist_col_residual_col = np.histogram2d(intersection_x_local,
                                                               difference_local[:, 0],
                                                               bins=(200, 800),
                                                               range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_column - 1 * std_column, mean_column + 1 * std_column)))

                        # Row residual agains row position
                        hist_row_residual_row = np.histogram2d(intersection_y_local,
                                                               difference_local[:, 1],
                                                               bins=(200, 800),
                                                               range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_row - 1 * std_row, mean_row + 1 * std_row)))

                        # Row residual agains column position
                        hist_col_residual_row = np.histogram2d(intersection_x_local,
                                                               difference_local[:, 1],
                                                               bins=(200, 800),
                                                               range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_row - 1 * std_row, mean_row + 1 * std_row)))

                        # Column residual agains row position
                        hist_row_residual_col = np.histogram2d(intersection_y_local,
                                                               difference_local[:, 0],
                                                               bins=(200, 800),
                                                               range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_column - 1 * std_column, mean_column + 1 * std_column)))
                    else:
                        # X residual agains x position
                        hist_x_residual_x += np.histogram2d(intersection_x,
                                                            difference[:, 0],
                                                            bins=(200, 800),
                                                            range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_x - 1 * std_x, mean_x + 1 * std_x)))

                        # Y residual agains y position
                        hist_y_residual_y += np.histogram2d(intersection_y,
                                                            difference[:, 1],
                                                            bins=(200, 800),
                                                            range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_y - 1 * std_y, mean_y + 1 * std_y)))

                        # Y residual agains x position
                        hist_x_residual_y += np.histogram2d(intersection_x,
                                                            difference[:, 1],
                                                            bins=(200, 800),
                                                            range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_y - 1 * std_y, mean_y + 1 * std_y)))

                        # X residual agains y position
                        hist_y_residual_x += np.histogram2d(intersection_y,
                                                            difference[:, 0],
                                                            bins=(200, 800),
                                                            range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_x - 1 * std_x, mean_x + 1 * std_x)))

                        # Column residual agains column position
                        hist_col_residual_col += np.histogram2d(intersection_x_local,
                                                                difference_local[:, 0],
                                                                bins=(200, 800),
                                                                range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_column - 1 * std_column, mean_column + 1 * std_column)))

                        # Row residual agains row position
                        hist_row_residual_row += np.histogram2d(intersection_y_local,
                                                                difference_local[:, 1],
                                                                bins=(200, 800),
                                                                range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_row - 1 * std_row, mean_row + 1 * std_row)))

                        # Row residual agains column position
                        hist_col_residual_row += np.histogram2d(intersection_x_local,
                                                                difference_local[:, 1],
                                                                bins=(200, 800),
                                                                range=((0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0]), (mean_row - 1 * std_row, mean_row + 1 * std_row)))

                        # Column residual agains row position
                        hist_row_residual_col += np.histogram2d(intersection_y_local,
                                                                difference_local[:, 0],
                                                                bins=(200, 800),
                                                                range=((0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1]), (mean_column - 1 * std_column, mean_column + 1 * std_column)))

                    residuals.append(std_column)
                    residuals.append(std_row)

                logging.debug('Store residual histograms')

                # Global residuals
                out_res_x = out_file_h5.createCArray(out_file_h5.root,
                                                     name='ResidualsX_DUT%d' % (actual_dut),
                                                     title='Residual distribution in x direction for DUT %d ' % (actual_dut),
                                                     atom=tb.Atom.from_dtype(hist_residual_x[0].dtype),
                                                     shape=hist_residual_x[0].shape,
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_x.attrs.x_edges = (hist_residual_x[1][0], hist_residual_x[1][-1])
                out_res_x[:] = hist_residual_x[0]

                out_res_y = out_file_h5.createCArray(out_file_h5.root,
                                                     name='ResidualsY_DUT%d' % (actual_dut),
                                                     title='Residual distribution in y direction for DUT %d ' % (actual_dut),
                                                     atom=tb.Atom.from_dtype(hist_residual_y[0].dtype),
                                                     shape=hist_residual_y[0].shape,
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_y.attrs.x_edges = (hist_residual_y[1][0], hist_residual_y[1][-1])
                out_res_y[:] = hist_residual_y[0]

                out_x_res_x = out_file_h5.createCArray(out_file_h5.root,
                                                       name='XResidualsX_DUT%d' % (actual_dut),
                                                       title='Residual distribution in x direction as a function of the x position for DUT %d ' % (actual_dut),
                                                       atom=tb.Atom.from_dtype(hist_x_residual_x[0].dtype),
                                                       shape=hist_x_residual_x[0].shape,
                                                       filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_x.attrs.x_edges = (hist_x_residual_x[1][0], hist_x_residual_x[1][-1])
                out_x_res_x.attrs.y_edges = (hist_x_residual_x[2][0], hist_x_residual_x[2][-1])
                out_x_res_x[:] = hist_x_residual_x[0]

                out_y_res_y = out_file_h5.createCArray(out_file_h5.root,
                                                       name='YResidualsY_DUT%d' % (actual_dut),
                                                       title='Residual distribution in y direction as a function of the y position for DUT %d ' % (actual_dut),
                                                       atom=tb.Atom.from_dtype(hist_y_residual_y[0].dtype),
                                                       shape=hist_y_residual_y[0].shape,
                                                       filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_y.attrs.x_edges = (hist_y_residual_y[1][0], hist_y_residual_y[1][-1])
                out_y_res_y.attrs.y_edges = (hist_y_residual_y[2][0], hist_y_residual_y[2][-1])
                out_y_res_y[:] = hist_y_residual_y[0]

                out_x_res_y = out_file_h5.createCArray(out_file_h5.root,
                                                       name='XResidualsY_DUT%d' % (actual_dut),
                                                       title='Residual distribution in y direction as a function of the x position for DUT %d ' % (actual_dut),
                                                       atom=tb.Atom.from_dtype(hist_x_residual_y[0].dtype),
                                                       shape=hist_x_residual_x[0].shape,
                                                       filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_y.attrs.x_edges = (hist_x_residual_y[1][0], hist_x_residual_y[1][-1])
                out_x_res_y.attrs.y_edges = (hist_x_residual_y[2][0], hist_x_residual_y[2][-1])
                out_x_res_y[:] = hist_x_residual_y[0]

                out_y_res_x = out_file_h5.createCArray(out_file_h5.root,
                                                       name='YResidualsX_DUT%d' % (actual_dut),
                                                       title='Residual distribution in x direction as a function of the y position for DUT %d ' % (actual_dut),
                                                       atom=tb.Atom.from_dtype(hist_y_residual_x[0].dtype),
                                                       shape=hist_y_residual_x[0].shape,
                                                       filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_x.attrs.x_edges = (hist_y_residual_x[1][0], hist_y_residual_x[1][-1])
                out_y_res_x.attrs.y_edges = (hist_y_residual_x[2][0], hist_y_residual_x[2][-1])
                out_y_res_x[:] = hist_y_residual_x[0]

                # Local residuals
                out_res_col = out_file_h5.createCArray(out_file_h5.root,
                                                       name='ResidualsColumn_DUT%d' % (actual_dut),
                                                       title='Residual distribution in column direction for DUT %d ' % (actual_dut),
                                                       atom=tb.Atom.from_dtype(hist_residual_col[0].dtype),
                                                       shape=hist_residual_col[0].shape,
                                                       filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_col.attrs.x_edges = (hist_residual_col[1][0], hist_residual_col[1][-1])
                out_res_col[:] = hist_residual_col[0]

                out_res_row = out_file_h5.createCArray(out_file_h5.root,
                                                       name='ResidualsRow_DUT%d' % (actual_dut),
                                                       title='Residual distribution in row direction for DUT %d ' % (actual_dut),
                                                       atom=tb.Atom.from_dtype(hist_residual_row[0].dtype),
                                                       shape=hist_residual_row[0].shape,
                                                       filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_row.attrs.x_edges = (hist_residual_row[1][0], hist_residual_row[1][-1])
                out_res_row[:] = hist_residual_row[0]

                out_col_res_col = out_file_h5.createCArray(out_file_h5.root,
                                                           name='ColumnResidualsCol_DUT%d' % (actual_dut),
                                                           title='Residual distribution in column direction as a function of the column position for DUT %d ' % (actual_dut),
                                                           atom=tb.Atom.from_dtype(hist_col_residual_col[0].dtype),
                                                           shape=hist_col_residual_col[0].shape,
                                                           filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_col.attrs.x_edges = (hist_col_residual_col[1][0], hist_col_residual_col[1][-1])
                out_col_res_col.attrs.y_edges = (hist_col_residual_col[2][0], hist_col_residual_col[2][-1])
                out_col_res_col[:] = hist_col_residual_col[0]

                out_row_res_row = out_file_h5.createCArray(out_file_h5.root,
                                                           name='RowResidualsRow_DUT%d' % (actual_dut),
                                                           title='Residual distribution in row direction as a function of the row position for DUT %d ' % (actual_dut),
                                                           atom=tb.Atom.from_dtype(hist_row_residual_row[0].dtype),
                                                           shape=hist_row_residual_row[0].shape,
                                                           filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_row.attrs.x_edges = (hist_row_residual_row[1][0], hist_row_residual_row[1][-1])
                out_row_res_row.attrs.y_edges = (hist_row_residual_row[2][0], hist_row_residual_row[2][-1])
                out_row_res_row[:] = hist_row_residual_row[0]

                out_col_res_row = out_file_h5.createCArray(out_file_h5.root,
                                                           name='ColumnResidualsRow_DUT%d' % (actual_dut),
                                                           title='Residual distribution in row direction as a function of the column position for DUT %d ' % (actual_dut),
                                                           atom=tb.Atom.from_dtype(hist_col_residual_row[0].dtype),
                                                           shape=hist_col_residual_col[0].shape,
                                                           filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_row.attrs.x_edges = (hist_col_residual_row[1][0], hist_col_residual_row[1][-1])
                out_col_res_row.attrs.y_edges = (hist_col_residual_row[2][0], hist_col_residual_row[2][-1])
                out_col_res_row[:] = hist_col_residual_row[0]

                out_row_res_col = out_file_h5.createCArray(out_file_h5.root,
                                                           name='RowResidualsColumn_DUT%d' % (actual_dut),
                                                           title='Residual distribution in column direction as a function of the row position for DUT %d ' % (actual_dut),
                                                           atom=tb.Atom.from_dtype(hist_row_residual_col[0].dtype),
                                                           shape=hist_row_residual_col[0].shape,
                                                           filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_col.attrs.x_edges = (hist_row_residual_col[1][0], hist_row_residual_col[1][-1])
                out_row_res_col.attrs.y_edges = (hist_row_residual_col[2][0], hist_row_residual_col[2][-1])
                out_row_res_col[:] = hist_row_residual_col[0]

                # Create plots
                if output_fig:
                    # Global residuals
                    logging.debug('Create residual plots')
                    coeff, var_matrix = None, None
                    try:
                        coeff, var_matrix = curve_fit(analysis_utils.gauss, hist_residual_x[1][:-1], hist_residual_x[0], p0=[np.amax(hist_residual_x[0]), mean_x, std_x])
                    except:  # Fit error
                        pass

                    plot_utils.plot_residuals(histogram=hist_residual_x,
                                              fit=coeff,
                                              fit_errors=var_matrix,
                                              title='Residuals for DUT %d' % actual_dut,
                                              x_label='X residual [um]',
                                              output_fig=output_fig)

                    coeff, var_matrix = None, None
                    try:
                        coeff, var_matrix = curve_fit(analysis_utils.gauss, hist_residual_y[1][:-1], hist_residual_y[0], p0=[np.amax(hist_residual_y[0]), mean_y, std_y])
                    except:  # Fit error
                        pass

                    plot_utils.plot_residuals(histogram=hist_residual_y,
                                              fit=coeff,
                                              fit_errors=var_matrix,
                                              title='Residuals for DUT %d' % actual_dut,
                                              x_label='Y residual [um]',
                                              output_fig=output_fig)

                    def get_median(array, values):  # Calculate the median of a 2D histogram along axis=1 with given values
                        def find_nearest(array, value):
                            return (np.abs(array - value[:, np.newaxis])).argmin(axis=1)
                        cumsum = np.cumsum(array, axis=1)
                        idx = find_nearest(cumsum, np.max(cumsum, axis=1) / 2.)
                        return values[idx]

                    x = np.linspace(hist_x_residual_x[1][0], hist_x_residual_x[1][-1], num=hist_x_residual_x[0].shape[0])
                    values = np.linspace(hist_x_residual_x[2][0], hist_x_residual_x[2][-1], num=hist_x_residual_x[0].shape[1])
                    y = get_median(hist_x_residual_x[0], values)
                    plot_utils.plot_position_residuals(hist_x_residual_x,
                                                       x,
                                                       y,
                                                       x_label='X position [um]',
                                                       y_label='X residual [um]',
                                                       output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1], hist_y_residual_y[0].shape[0])
                    values = np.linspace(mean_y - 1 * std_y, mean_y + 1 * std_y, 800)
                    y = get_median(hist_y_residual_y[0], values)
                    plot_utils.plot_position_residuals(hist_y_residual_y,
                                                       x,
                                                       y,
                                                       x_label='Y position [um]',
                                                       y_label='Y residual [um]',
                                                       output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0], hist_x_residual_y[0].shape[0])
                    values = np.linspace(mean_y - 1 * std_y, mean_y + 1 * std_y, 800)
                    y = get_median(hist_x_residual_y[0], values)
                    plot_utils.plot_position_residuals(hist_x_residual_y,
                                                       x,
                                                       y,
                                                       x_label='X position [um]',
                                                       y_label='Y residual [um]',
                                                       output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1], hist_y_residual_x[0].shape[0])
                    values = np.linspace(mean_x - 1 * std_x, mean_x + 1 * std_x, 800)
                    y = get_median(hist_y_residual_x[0], values)
                    plot_utils.plot_position_residuals(hist_y_residual_x,
                                                       x,
                                                       y,
                                                       x_label='X position [um]',
                                                       y_label='Y residual [um]',
                                                       output_fig=output_fig)

                    # Local residuals
                    coeff, var_matrix = None, None
                    try:
                        coeff, var_matrix = curve_fit(analysis_utils.gauss, hist_residual_col[1][:-1], hist_residual_col[0], p0=[np.amax(hist_residual_col[0]), mean_column, std_column])
                    except:  # Fit error
                        pass

                    plot_utils.plot_residuals(histogram=hist_residual_col,
                                              fit=coeff,
                                              fit_errors=var_matrix,
                                              title='Residuals for DUT %d' % actual_dut,
                                              x_label='Column residual [um]',
                                              output_fig=output_fig)

                    coeff, var_matrix = None, None
                    try:
                        coeff, var_matrix = curve_fit(analysis_utils.gauss, hist_residual_row[1][:-1], hist_residual_row[0], p0=[np.amax(hist_residual_row[0]), mean_row, std_row])
                    except:  # Fit error
                        pass

                    plot_utils.plot_residuals(histogram=hist_residual_row,
                                              fit=coeff,
                                              fit_errors=var_matrix,
                                              title='Residuals for DUT %d' % actual_dut,
                                              x_label='Row residual [um]',
                                              output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0], hist_col_residual_col[0].shape[0])
                    values = np.linspace(mean_column - 1 * std_column, mean_column + 1 * std_column, 800)
                    y = get_median(hist_col_residual_col[0], values)
                    plot_utils.plot_position_residuals(hist_col_residual_col,
                                                       x,
                                                       y,
                                                       x_label='Column position [um]',
                                                       y_label='Column residual [um]',
                                                       output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1], hist_row_residual_row[0].shape[0])
                    values = np.linspace(mean_row - 1 * std_row, mean_row + 1 * std_row, 800)
                    y = get_median(hist_row_residual_row[0], values)
                    plot_utils.plot_position_residuals(hist_row_residual_row,
                                                       x,
                                                       y,
                                                       x_label='Row position [um]',
                                                       y_label='Row residual [um]',
                                                       output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][0] * pixel_size[actual_dut][0], hist_col_residual_row[0].shape[0])
                    values = np.linspace(mean_row - 1 * std_row, mean_row + 1 * std_row, 800)
                    y = get_median(hist_col_residual_row[0], values)
                    plot_utils.plot_position_residuals(hist_col_residual_row,
                                                       x,
                                                       y,
                                                       x_label='Column position [um]',
                                                       y_label='Row residual [um]',
                                                       output_fig=output_fig)

                    x = np.linspace(0, n_pixels[actual_dut][1] * pixel_size[actual_dut][1], hist_row_residual_col[0].shape[0])
                    values = np.linspace(mean_column - 1 * std_column, mean_column + 1 * std_column, 800)
                    y = get_median(hist_row_residual_col[0], values)
                    plot_utils.plot_position_residuals(hist_row_residual_col,
                                                       x,
                                                       y,
                                                       x_label='Row position [um]',
                                                       y_label='Column residual [um]',
                                                       output_fig=output_fig)

    if output_fig:
        output_fig.close()

    return residuals


def calculate_efficiency(input_tracks_file, input_alignment_file, output_pdf, bin_size, minimum_track_density, max_distance=500, sensor_size=None, use_duts=None, max_chi2=None, force_prealignment=False, cut_distance=None, col_range=None, row_range=None, output_file=None):
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
    minimum_track_density : int
        minimum track density required to consider bin for efficiency calculation
    sensor_size : iterable
        size of the used sensor in um. Give one tuple (x, y) for every plane or list of tuples for different planes
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
    '''
    logging.info('=== Calculate efficiency ===')

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        if not use_prealignment:
            try:
                alignment = in_file_h5.root.Alignment[:]
                logging.info('Use alignment data')
            except tb.exceptions.NodeError:
                use_prealignment = True
                logging.info('Use prealignment data')

    with PdfPages(output_pdf) as output_fig:
        efficiencies = []
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for index, node in enumerate(in_file_h5.root):
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Calculate efficiency for DUT %d', actual_dut)
                track_array = node[:]

                # Get pixel and bin sizes for calculations and plotting
                # Allow different sensor sizes for every plane
                if not sensor_size:
                    dimensions = (np.amax(track_array['x_dut_%d' % actual_dut]), np.amax(track_array['y_dut_%d' % actual_dut]))
                else:
                    dimensions = [sensor_size, ] if not isinstance(sensor_size, list) else sensor_size
                    if len(dimensions) == 1:
                        dimensions = dimensions[0]
                    else:
                        dimensions = dimensions[index]

                # Allow different bin_sizes for every plane
                bin_size = [bin_size, ] if not isinstance(bin_size, list) else bin_size
                if len(bin_size) != 1:
                    actual_bin_size_x = bin_size[index][0]
                    actual_bin_size_y = bin_size[index][1]
                else:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]

                n_bin_x = dimensions[0] / actual_bin_size_x
                n_bin_y = dimensions[1] / actual_bin_size_y

                # Cut in Chi 2 of the track fit
                if max_chi2:
                    track_array = track_array[track_array['track_chi2'] <= max_chi2]

                # Transform the hits and track intersections into the local coordinate system (otherwise they are quite meaningless)
                if not use_prealignment:  # Use alignment with rotations
                    transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=alignment[actual_dut]['translation_x'],
                                                                                                 y=alignment[actual_dut]['translation_y'],
                                                                                                 z=alignment[actual_dut]['translation_z'],
                                                                                                 alpha=alignment[actual_dut]['alpha'],
                                                                                                 beta=alignment[actual_dut]['beta'],
                                                                                                 gamma=alignment[actual_dut]['gamma'])
                else:  # Prealignment only gives offsets and no rotations, thus apply the known offsets -> col/row/z position is x/y/z offset
                    transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=prealignment[actual_dut]['column_c0'],
                                                                                                 y=prealignment[actual_dut]['row_c0'],
                                                                                                 z=prealignment[actual_dut]['z'],
                                                                                                 alpha=0,
                                                                                                 beta=0,
                                                                                                 gamma=0)

                hit_x, hit_y, hit_z = geometry_utils.apply_transformation_matrix(x=track_array['x_dut_%d' % actual_dut],
                                                                                 y=track_array['y_dut_%d' % actual_dut],
                                                                                 z=track_array['z_dut_%d' % actual_dut],
                                                                                 transformation_matrix=transformation_matrix)

                intersection_x, intersection_y, intersection_z = geometry_utils.apply_transformation_matrix(x=track_array['offset_0'],
                                                                                                            y=track_array['offset_1'],
                                                                                                            z=track_array['offset_2'],
                                                                                                            transformation_matrix=transformation_matrix)

                intersection = np.column_stack((intersection_x, intersection_y, intersection_z))
                hits = np.column_stack((hit_x, hit_y, hit_z))

                selection = np.logical_or(track_array['x_dut_%d' % actual_dut] == 0., track_array['y_dut_%d' % actual_dut] == 0.)  # Only transform real hits
                hits[selection, 0] = 0.
                hits[selection, 1] = 0.

                if not np.allclose(hit_z[~selection], 0) or not np.allclose(intersection_z, 0):
                    raise RuntimeError('The transformation to the local coordinate system did not give all z = 0.')

                # Select hits from column row range (e.g. to supress edge pixels)
                col_range = [col_range, ] if not isinstance(col_range, list) else col_range
                row_range = [row_range, ] if not isinstance(row_range, list) else row_range
                if len(col_range) == 1:
                    index = 0
                if len(row_range) == 1:
                    index = 0
                if col_range[index] is not None:
                    selection = np.logical_and(intersection[:, 0] >= col_range[index][0], intersection[:, 0] <= col_range[index][1])  # Select real hits
                    hits, intersection = hits[selection], intersection[selection]
                if row_range[index] is not None:
                    selection = np.logical_and(intersection[:, 1] >= row_range[index][0], intersection[:, 1] <= row_range[index][1])  # Select real hits
                    hits, intersection = hits[selection], intersection[selection]

                events = track_array['event_number'][selection]
                print events

                # Calculate distance between track hit and DUT hit
                scale = np.square(np.array((1, 1, 0)))  # regard pixel size for calculating distances
                distance = np.sqrt(np.dot(np.square(intersection - hits), scale))  # array with distances between DUT hit and track hit for each event. Values in um

                col_row_distance = np.column_stack((hits[:, 0], hits[:, 1], distance))
                distance_array = np.histogramdd(col_row_distance, bins=(n_bin_x, n_bin_y, max_distance), range=[[0, dimensions[0]], [0, dimensions[1]], [0, max_distance]])[0]
                hit_hist, _, _ = np.histogram2d(hits[:, 0], hits[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])

                # Calculate distances between hit and intersection
                distance_mean_array = np.average(distance_array, axis=2, weights=range(0, max_distance)) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_mean_array = np.ma.masked_invalid(distance_mean_array)
                distance_max_array = np.amax(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_min_array = np.amin(distance_array, axis=2) * sum(range(0, max_distance)) / hit_hist.astype(np.float)
                distance_max_array = np.ma.masked_invalid(distance_max_array)
                distance_min_array = np.ma.masked_invalid(distance_min_array)

                # Calculate efficiency
                if cut_distance:  # Select intersections where hit is in given distance around track intersection
                    intersection_valid_hit = intersection[np.logical_and(np.logical_and(hits[:, 0] != 0, hits[:, 1] != 0), distance < cut_distance)]
                else:
                    intersection_valid_hit = intersection[np.logical_and(hits[:, 0] != 0, hits[:, 1] != 0)]

                track_density, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])
                track_density_with_DUT_hit, _, _ = np.histogram2d(intersection_valid_hit[:, 0], intersection_valid_hit[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])

                if np.all(track_density == 0):
                    logging.warning('No tracks on DUT %d, cannot calculate efficiency', actual_dut)
                    continue

                efficiency = np.zeros_like(track_density_with_DUT_hit)
                efficiency[track_density != 0] = track_density_with_DUT_hit[track_density != 0].astype(np.float) / track_density[track_density != 0].astype(np.float) * 100.

                efficiency = np.ma.array(efficiency, mask=track_density < minimum_track_density)

                plot_utils.efficiency_plots(distance_min_array, distance_max_array, distance_mean_array, hit_hist, track_density, track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, plot_range=dimensions, cut_distance=cut_distance, output_fig=output_fig)

                logging.info('Efficiency =  %1.4f +- %1.4f', np.ma.mean(efficiency), np.ma.std(efficiency))
                efficiencies.append(np.ma.mean(efficiency))

                if output_file:
                    with tb.open_file(output_file, 'a') as out_file_h5:
                        actual_dut_folder = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)
                        out_efficiency = out_file_h5.createCArray(actual_dut_folder, name='Efficiency', title='Efficiency map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.dtype), shape=efficiency.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                        out_efficiency_mask = out_file_h5.createCArray(actual_dut_folder, name='Efficiency_mask', title='Masked pixel map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.mask.dtype), shape=efficiency.mask.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
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
    return efficiencies
