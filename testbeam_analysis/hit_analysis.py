''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import os.path

import tables as tb
import numpy as np
from scipy.ndimage import median_filter

from pixel_clusterizer.clusterizer import HitClusterizer
from testbeam_analysis import analysis_utils
from plot_utils import plot_noisy_pixels


def remove_noisy_pixels(input_raw_data_file, n_pixel, pixel_size=None, threshold=10.0, chunk_size=1000000):
    '''Removes noisy pixel from the data file containing the hit table.
    The hit table is read in chunks and for each chunk the noisy pixel are determined and removed.

    To call this function on 8 cores in parallel with chunk_size=1000000 the following RAM is needed:
    11 byte * 8 * 1000000 = 88 Mb

    Parameters
    ----------
    input_raw_data_file : string
        Input PyTables raw data file.
    n_pixel : tuple
        Total number of pixels per column and row.
    pixel_size : tuple
        Pixel dimension for column and row. If None, assuming square pixels.
    threshold : float
        The threshold for pixel masking. The threshold is given in units of sigma of the pixel noise (background subtracted).
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Removing noisy pixel in %s ===', input_raw_data_file)
    occupancy = None
    # calculating occupancy array
    with tb.open_file(input_raw_data_file, 'r') as input_file_h5:
        for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
            col, row = hits['column'], hits['row']
            chunk_occ = analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)
            if occupancy is None:
                occupancy = chunk_occ
            else:
                occupancy = occupancy + chunk_occ

    # run median filter across data, assuming 0 filling past the edges
    blurred = median_filter(occupancy.astype(np.int32), size=2, mode='constant', cval=0.0)
    difference = np.ma.masked_array(occupancy - blurred)

    std = np.ma.std(difference)
    abs_occ_threshold = threshold * std
    occupancy = np.ma.masked_where(difference > abs_occ_threshold, occupancy)
    logging.info('Removed a total of %d hot pixel at threshold %.1f in %s', np.ma.count_masked(occupancy), threshold, input_raw_data_file)

    # generate tuple col / row array of hot pixels, do not use getmask()
    noisy_pixels_mask = np.ma.getmaskarray(occupancy)
    # generate pair of col / row arrays
    noisy_pixels = np.nonzero(noisy_pixels_mask)
    # check for any noisy pixels
    if noisy_pixels[0].shape[0] != 0:
        # map 2d array (col, row) to 1d array to increase selection speed
        noisy_pixels_1d = (noisy_pixels[0] + 1) * n_pixel[1] + (noisy_pixels[1] + 1)
    else:
        noisy_pixels_1d = []

    # storing putput files
    with tb.open_file(input_raw_data_file, 'r') as input_file_h5:
        output_raw_data_file = os.path.splitext(input_raw_data_file)[0] + '_noisy_pixels.h5'
        with tb.open_file(output_raw_data_file, 'w') as out_file_h5:
            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=input_file_h5.root.Hits.dtype, title='Selected not noisy hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
                # Select not noisy pixel
                hits_1d = hits['column'].astype(np.uint32) * n_pixel[1] + hits['row']  # change dtype to fit new number
                hits = hits[np.in1d(hits_1d, noisy_pixels_1d, invert=True)]

                hit_table_out.append(hits)

            logging.info('Reducing data by a factor of %.2f in %s', hit_table_out.nrows / input_file_h5.root.Hits.nrows, out_file_h5.filename)

    output_pdf_file = os.path.splitext(input_raw_data_file)[0] + '_noisy_pixels.pdf'
    plot_noisy_pixels(occupancy, output_pdf_file, pixel_size)

# testing output file
#     occupancy = None
#     with tb.open_file(data_file[:-3] + '_hot_pixel.h5', 'r') as input_file_h5:
#         for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
#             col, row = hits['column'], hits['row']
#             chunk_occ = analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)
#             if occupancy is None:
#                 occupancy = chunk_occ
#             else:
#                 occupancy = occupancy + chunk_occ
#
#     occupancy = np.ma.masked_where(occupancy == 0, occupancy)
#     plt.figure()
#     plt.imshow(occupancy, cmap=cmap, norm=norm, interpolation='none', origin='lower', clim=(0, np.percentile(occupancy, 99)))
#     plt.show()


def remove_noisy_pixels_wrapper(args):
    return remove_noisy_pixels(**args)


def cluster_hits_wrapper(args):
    return cluster_hits(**args)


def cluster_hits(data_file, max_x_distance=3, max_y_distance=3, max_time_distance=2, max_cluster_hits=1000, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    data_file : pytables file
    output_file : pytables file
    '''

    logging.info('=== Cluster hits in %s ===', data_file)

    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_cluster.h5', 'w') as output_file_h5:
            # create clusterizer object
            clusterizer = HitClusterizer()
            clusterizer.set_max_hits(chunk_size)
            clusterizer.set_max_cluster_hits(max_cluster_hits)

            # Set clusterzier settings
            clusterizer.create_cluster_hit_info_array(False)  # do not create cluster infos for hits
            clusterizer.set_x_cluster_distance(max_x_distance)  # cluster distance in columns
            clusterizer.set_y_cluster_distance(max_y_distance)  # cluster distance in rows
            clusterizer.set_frame_cluster_distance(max_time_distance)  # cluster distance in time frames

            # Output data
            cluster_table_description = np.dtype([('event_number', '<i8'),
                                                  ('ID', '<u2'),
                                                  ('n_hits', '<u2'),
                                                  ('charge', 'f4'),
                                                  ('seed_column', '<u2'),
                                                  ('seed_row', '<u2'),
                                                  ('mean_column', 'f4'),
                                                  ('mean_row', 'f4')])
            cluster_table_out = output_file_h5.createTable(output_file_h5.root, name='Cluster', description=cluster_table_description, title='Clustered hits', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

            for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size, try_speedup=False):
                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. The hits cannot be used like this!')
                __, cluster = clusterizer.cluster_hits(hits)  # Cluster hits
                if not np.all(np.diff(cluster['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. The cluster cannot be used like this!')
                cluster_table_out.append(cluster)
