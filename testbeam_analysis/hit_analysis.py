''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import collections

import tables as tb
import numpy as np
from scipy.ndimage import median_filter

from pixel_clusterizer.clusterizer import HitClusterizer
from testbeam_analysis import analysis_utils
from plot_utils import plot_noisy_pixel


def remove_noisy_pixel(data_file, n_pixel, threshold=(20, 5, 2), chunk_size=1000000):
    '''Removes noisy pixel from the data file containing the hit table.
    The hit table is read in chunks and for each chunk the noisy pixel are determined and removed.

    To call this function on 8 cores in parallel with chunk_size=1000000 the following RAM is needed:
    11 byte * 8 * 1000000 = 88 Mb

    Parameters
    ----------
    data_file : pytables file
    threshold : number
        The threshold when the pixel is removed given in sigma distance from the median occupancy.
    '''
    logging.info('=== Removing noisy pixel in %s ===', data_file)
    occupancy = None
    with tb.open_file(data_file, 'r') as input_file_h5:
        for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
            col, row = hits['column'], hits['row']
            chunk_occ = analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)
            if occupancy is None:
                occupancy = chunk_occ
            else:
                occupancy = occupancy + chunk_occ

    if not isinstance(threshold, collections.Iterable):
        threshold = [threshold]
    blurred = median_filter(occupancy.astype(np.int32), size=2, mode='constant')
    difference = occupancy - blurred

    difference = np.ma.masked_array(difference)
    for thr in threshold:
        std = np.ma.std(difference)
        threshold = thr * std
        occupancy = np.ma.masked_where(difference > threshold, occupancy)
        logging.info('Removed a total of %d hot pixel at threshold %.1f in %s', np.ma.count_masked(occupancy), thr, data_file)
        difference = np.ma.masked_array(difference, mask=np.ma.getmask(occupancy))

    hot_pixel = np.nonzero(np.ma.getmask(occupancy))
    noisy_pix_1d = (hot_pixel[0] + 1) * n_pixel[1] + (hot_pixel[1] + 1)  # map 2d array (col, row) to 1d array to increase selection speed

    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_hot_pixel.h5', 'w') as out_file_h5:
            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=input_file_h5.root.Hits.dtype, title='Selected not noisy hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
                # Select not noisy pixel
                hits_1d = hits['column'].astype(np.uint32) * n_pixel[1] + hits['row']  # change dtype to fit new number
                hits = hits[np.in1d(hits_1d, noisy_pix_1d, invert=True)]

                hit_table_out.append(hits)

            logging.info('Reducing data by a factor of %.2f in %s', hit_table_out.nrows / input_file_h5.root.Hits.nrows, out_file_h5.filename)

    plot_noisy_pixel(occupancy, data_file[:-3] + '_hot_pixel.pdf')

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
#     plt.imshow(occupancy, cmap=cmap, norm=norm, interpolation='none', origin='lower', clim=(0, 2 * np.ma.median(occupancy)))
#     plt.show()


def remove_noisy_pixel_wrapper(args):
    return remove_noisy_pixel(**args)


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
