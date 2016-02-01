''' All functions acting on the hits of one DUT are listed here'''

import logging
import tables as tb
import numpy as np
from pixel_clusterizer.clusterizer import HitClusterizer

from testbeam_analysis import analysis_utils
from testbeam_analysis import plot_utils


def remove_noisy_pixels(data_file, threshold=6.):
    '''Removes noisy pixels from the data file containing the hit table.

    Parameters
    ----------
    data_file : pytables file
    threshold : number
        The threshold when the pixel is removed given in sigma distance from the median occupancy.
    '''
    logging.info('Remove noisy pixels in %s', data_file)
    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_hot_pixel.h5', 'w') as out_file_h5:
            hits = input_file_h5.root.Hits[:]
            col, row = hits['column'], hits['row']
            max_row = np.amax(row)
            occupancy = analysis_utils.hist_2d_index(col - 1, row - 1, shape=(np.amax(col), max_row))
            noisy_pixels = np.where(occupancy > np.median(occupancy) + np.std(occupancy) * threshold)
            plot_utils.plot_noisy_pixel(occupancy, noisy_pixels, threshold, filename=data_file[:-3] + '_hot_pixel.pdf')
            logging.info('Remove %d hot pixels in %s', noisy_pixels[0].shape[0], data_file)

            # Select not noisy pixels
            noisy_pix_1d = (noisy_pixels[0] + 1) * max_row + (noisy_pixels[1] + 1)  # map 2d array (col, row) to 1d array to increase selection speed
            hits_1d = hits['column'].astype(np.uint32) * max_row + hits['row']  # astype needed, otherwise silently assuming np.uint16 (VERY BAD NUMPY!)
            hits = hits[np.in1d(hits_1d, noisy_pix_1d, invert=True)]

            hit_table_out = out_file_h5.createTable(out_file_h5.root, name='Hits', description=hits.dtype, title='Selected not noisy hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hit_table_out.append(hits)


def cluster_hits_wrapper(args):
    return cluster_hits(*args)


def cluster_hits(data_file, max_x_distance=3, max_y_distance=3, max_time_distance=2, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    data_file : pytables file
    output_file : pytables file
    '''

    logging.info('Cluster hits in %s', data_file)

    with tb.open_file(data_file, 'r') as input_file_h5:
        with tb.open_file(data_file[:-3] + '_cluster.h5', 'w') as output_file_h5:
            # create clusterizer object
            clusterizer = HitClusterizer()
            clusterizer.set_max_hits(chunk_size)

            # Set clusterzier settings
            clusterizer.create_cluster_hit_info_array(False)  # do not create cluster infos for hits
            clusterizer.set_x_cluster_distance(max_x_distance)  # cluster distance in columns
            clusterizer.set_y_cluster_distance(max_y_distance)  # cluster distance in rows
            clusterizer.set_frame_cluster_distance(max_time_distance)   # cluster distance in time frames

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

            for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
                clusterizer.cluster_hits(hits)  # Cluster hits
                cluster = clusterizer.get_cluster()
                cluster_table_out.append(cluster)
