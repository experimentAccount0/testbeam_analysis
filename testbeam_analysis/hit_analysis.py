''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import os.path
import re

import tables as tb
import numpy as np
from scipy.ndimage import median_filter

from pixel_clusterizer.clusterizer import HitClusterizer
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_size


def generate_pixel_mask(input_hits_file, n_pixel, pixel_mask_name="NoisyPixelMask", output_mask_file=None, pixel_size=None, threshold=10.0, filter_size=3, dut_name=None, plot=True, chunk_size=1000000):
    '''Generating pixel mask from the hit table.

    Parameters
    ----------
    input_hits_file : string
        File name of the hit table.
    n_pixel : tuple
        Tuple of the total number of pixels (column/row).
    pixel_mask_name : string
        Name of the node containing the mask inside the output file.
    output_mask_file : string
        File name of the output mask file.
    pixel_size : tuple
        Tuple of the pixel size (column/row). If None, assuming square pixels.
    threshold : float
        The threshold for pixel masking. The threshold is given in units of
        sigma of the pixel noise (background subtracted). The lower the value
        the more pixels are masked.
    filter_size : scalar or tuple
        Adjust the median filter size by giving the number of columns and rows.
        The higher the value the more the background is smoothed and more
        pixels are masked.
    dut_name : string
        Name of the DUT. If None, file name of the hit table will be printed.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Generating %s for %s ===', ' '.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)), input_hits_file)

    if output_mask_file is None:
        output_mask_file = os.path.splitext(input_hits_file)[0] + '_' + '_'.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)) + '.h5'

    occupancy = None
    # Calculating occupancy array
    with tb.open_file(input_hits_file, 'r') as input_file_h5:
        for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
            col, row = hits['column'], hits['row']
            chunk_occ = analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)
            if occupancy is None:
                occupancy = chunk_occ
            else:
                occupancy = occupancy + chunk_occ

    # Run median filter across data, assuming 0 filling past the edges to get expected occupancy
    blurred = median_filter(occupancy.astype(np.int32), size=filter_size, mode='constant', cval=0.0)
    # Spot noisy pixels maxima by substracting expected occupancy
    difference = np.ma.masked_array(occupancy - blurred)
    std = np.ma.std(difference)
    abs_occ_threshold = threshold * std
    occupancy = np.ma.masked_where(difference > abs_occ_threshold, occupancy)
    logging.info('Masked %d pixels at threshold %.1f in %s', np.ma.count_masked(occupancy), threshold, input_hits_file)
    # Generate tuple col / row array of hot pixels, do not use getmask()
    pixel_mask = np.ma.getmaskarray(occupancy)

    with tb.open_file(output_mask_file, 'w') as out_file_h5:
        # Create occupancy array without masking pixels
        occupancy_array_table = out_file_h5.create_carray(out_file_h5.root, name='HistOcc', title='Occupancy Histogram', atom=tb.Atom.from_dtype(occupancy.dtype), shape=occupancy.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        occupancy_array_table[:] = np.ma.getdata(occupancy)

        # Create masked pixels array
        masked_pixel_table = out_file_h5.create_carray(out_file_h5.root, name=pixel_mask_name, title='Pixel Mask', atom=tb.Atom.from_dtype(pixel_mask.dtype), shape=pixel_mask.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        masked_pixel_table[:] = pixel_mask

    if plot:
        plot_masked_pixels(input_mask_file=output_mask_file, pixel_size=pixel_size, dut_name=dut_name)

    return output_mask_file


def cluster_hits(input_hits_file, output_cluster_file=None, create_cluster_hits_table=False, input_disabled_pixel_mask_file=None, input_noisy_pixel_mask_file=None, min_hit_charge=0, max_hit_charge=None, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, dut_name=None, plot=True, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    input_hits_file : string
        Filename of the input hits file.
    output_cluster_file : string
        Filename of the output cluster file. If None, the filename will be derived from the input hits file.
    create_cluster_hits_table : bool
        If True, additionally create cluster hits table.
    input_disabled_pixel_mask_file : string
        Filename of the input disabled mask file.
    input_noisy_pixel_mask_file : string
        Filename of the input disabled mask file.
    min_hit_charge : uint
        Minimum hit charge. Minimum possible hit charge must be given in order to correcly calculate the cluster coordinates.
    max_hit_charge : uint
        Maximum hit charge. Hits wit charge above the limit will be ignored.
    column_cluster_distance : uint
        Maximum column distance between hist so that they are assigned to the same cluster. Value of 0 effectively disables the clusterizer in column direction.
    row_cluster_distance : uint
        Maximum row distance between hist so that they are assigned to the same cluster. Value of 0 effectively disables the clusterizer in row direction.
    frame_cluster_distance : uint
        Sometimes an event has additional timing information (e.g. bunch crossing ID, frame ID). Value of 0 effectively disables the clusterization in time.
    dut_name : string
        Name of the DUT. If None, filename of the output cluster file will be used.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Clustering hits in %s ===', input_hits_file)

    if output_cluster_file is None:
        output_cluster_file = os.path.splitext(input_hits_file)[0] + '_clustered.h5'

    # Calculate the size in col/row for each cluster
    def end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        min_col = hits[cluster_hit_indices[0]].column
        max_col = hits[cluster_hit_indices[0]].column
        min_row = hits[cluster_hit_indices[0]].row
        max_row = hits[cluster_hit_indices[0]].row
        for i in cluster_hit_indices[1:]:
            if i < 0:  # Not used indeces = -1
                break
            if hits[i].column < min_col:
                min_col = hits[i].column
            if hits[i].column > max_col:
                max_col = hits[i].column
            if hits[i].row < min_row:
                min_row = hits[i].row
            if hits[i].row > max_row:
                max_row = hits[i].row
        clusters[cluster_index].err_column = max_col - min_col + 1
        clusters[cluster_index].err_row = max_row - min_row + 1

    # Adding number of clusters
    def end_of_event_function(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
        for i in range(start_event_cluster_index, stop_event_cluster_index):
            clusters[i]['n_cluster'] = hits["n_cluster"][start_event_hit_index]

    with tb.open_file(input_hits_file, 'r') as input_file_h5:
        with tb.open_file(output_cluster_file, 'w') as output_file_h5:
            if input_disabled_pixel_mask_file is not None:
                with tb.open_file(input_disabled_pixel_mask_file, 'r') as input_mask_file_h5:
                    disabled_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.DisabledPixelMask[:]))[0] + 1
                    input_mask_file_h5.root.DisabledPixelMask._f_copy(newparent=output_file_h5.root)
            else:
                disabled_pixels = None
            if input_noisy_pixel_mask_file is not None:
                with tb.open_file(input_noisy_pixel_mask_file, 'r') as input_mask_file_h5:
                    noisy_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.NoisyPixelMask[:]))[0] + 1
                    input_mask_file_h5.root.NoisyPixelMask._f_copy(newparent=output_file_h5.root)
            else:
                noisy_pixels = None

            hit_dtype = np.dtype([('event_number', '<i8'),
                                  ('frame', '<u1'),
                                  ('column', '<u2'),
                                  ('row', '<u2'),
                                  ('charge', '<u2'),  # TODO: change that
                                  ('cluster_ID', '<i4'),  # TODO: change that
                                  ('is_seed', '<u1'),
                                  ('cluster_size', '<u4'),
                                  ('n_cluster', '<u4')])
 
            cluster_dtype = np.dtype([('event_number', '<i8'),
                                      ('ID', '<u4'),
                                      ('n_hits', '<u4'),
                                      ('charge', '<f4'),
                                      ('seed_column', '<u2'),
                                      ('seed_row', '<u2'),
                                      ('mean_column', '<f8'),
                                      ('mean_row', '<f8')])

            clusterizer = HitClusterizer(column_cluster_distance=column_cluster_distance,
                                         row_cluster_distance=row_cluster_distance,
                                         frame_cluster_distance=frame_cluster_distance,
                                         min_hit_charge=min_hit_charge,
                                         max_hit_charge=max_hit_charge,
                                         hit_dtype=hit_dtype,
                                         cluster_dtype=cluster_dtype)
            clusterizer.add_cluster_field(description=('err_column', '<f4'))  # Add an additional field to hold the cluster size in x
            clusterizer.add_cluster_field(description=('err_row', '<f4'))  # Add an additional field to hold the cluster size in y
            clusterizer.add_cluster_field(description=('n_cluster', '<u4'))  # Adding additional field for number of clusters per event
            clusterizer.set_end_of_cluster_function(end_of_cluster_function)  # Set the new function to the clusterizer
            clusterizer.set_end_of_event_function(end_of_event_function)

            cluster_hits_table = None
            cluster_table = None
            for hits, _ in analysis_utils.data_aligned_at_events(input_file_h5.root.Hits, chunk_size=chunk_size):
                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. The hits cannot be used like this!')
                cluster_hits, clusters = clusterizer.cluster_hits(hits, noisy_pixels=noisy_pixels, disabled_pixels=disabled_pixels)  # Cluster hits
                if not np.all(np.diff(clusters['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. The cluster cannot be used like this!')
                # create cluster hits table dynamically
                if create_cluster_hits_table and cluster_hits_table is None:
                    cluster_hits_table = output_file_h5.create_table(output_file_h5.root, name='ClusterHits', description=cluster_hits.dtype, title='Cluster hits table', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                # create cluster table dynamically
                if cluster_table is None:
                    cluster_table = output_file_h5.create_table(output_file_h5.root, name='Cluster', description=clusters.dtype, title='Cluster table', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                if create_cluster_hits_table:
                    cluster_hits_table.append(cluster_hits)
                cluster_table.append(clusters)

    def get_eff_pitch(hist, cluster_size):
        ''' Effective pitch to describe the cluster
            size propability distribution

        hist : array like
            Histogram with cluster size distribution
        cluster_size : Cluster size to calculate the pitch for
        '''

        return np.sqrt(hight[int(cluster_size)].astype(np.float) / hight.sum())

    # Calculate cluster size histogram
    with tb.open_file(output_cluster_file, 'r') as input_file_h5:
        hight = None
        n_hits = 0
        n_clusters = input_file_h5.root.Cluster.nrows
        for start_index in range(0, n_clusters, chunk_size):
            cluster_n_hits = input_file_h5.root.Cluster[start_index:start_index + chunk_size]['n_hits']
            # calculate cluster size histogram
            if hight is None:
                max_cluster_size = np.amax(cluster_n_hits)
                hight = analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
            elif max_cluster_size < np.amax(cluster_n_hits):
                max_cluster_size = np.amax(cluster_n_hits)
                hight.resize(max_cluster_size + 1)
                hight += analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
            else:
                hight += analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
            n_hits += np.sum(cluster_n_hits)

    # Calculate cluster size histogram
    with tb.open_file(output_cluster_file, 'r+') as io_file_h5:
        for start_index in range(0, io_file_h5.root.Cluster.nrows, chunk_size):
            clusters = io_file_h5.root.Cluster[start_index:start_index + chunk_size]
            # Set errors for small clusters, where charge sharing enhances resolution
            for css in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                sel = np.logical_and(clusters['err_column'] == css[0], clusters['err_row'] == css[1])
                clusters['err_column'][sel] = get_eff_pitch(hist=hight, cluster_size=css[0]) / np.sqrt(12)
                clusters['err_row'][sel] = get_eff_pitch(hist=hight, cluster_size=css[1]) / np.sqrt(12)
            # Set errors for big clusters, where delta electrons reduce resolution
            sel = np.logical_or(clusters['err_column'] > 2, clusters['err_row'] > 2)
            clusters['err_column'][sel] = clusters['err_column'][sel] / np.sqrt(12)
            clusters['err_row'][sel] = clusters['err_row'][sel] / np.sqrt(12)
            io_file_h5.root.Cluster[start_index:start_index + chunk_size] = clusters

    if plot:
        plot_cluster_size(hight, n_hits, n_clusters, max_cluster_size,
                          dut_name=os.path.split(output_cluster_file)[1],
                          output_pdf_file=os.path.splitext(output_cluster_file)[0] + '_cluster_size.pdf')

    return output_cluster_file
