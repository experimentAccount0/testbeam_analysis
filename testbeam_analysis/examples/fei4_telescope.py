''' The FE-I4 telescope data example shows how to run a full analysis on data
    taken with a FE-I4 telescope.

    .. NOTE::
       Only prealignment is done here, since the telescope data is too coarse
       to profit from an aligment step. The data was recorded at DESY with
       pyBar. The telescope consists of 6 DUTs with ~ 2 cm distance between the
       planes. Only the first two and last two planes were taken here. The
       first and last plane were IBL n-in-n planar sensors and the 2 devices in
       the center 3D CNM/FBK sensors.

'''

import os
import inspect
import logging
from multiprocessing import Pool

from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis.tools import plot_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis():
    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')

    # The location of the data files, one file per DUT
    data_files = [(os.path.join(tests_data_folder, 'TestBeamData_FEI4_DUT%d' % i + '.h5')) for i in [0, 1, 4, 5]]  # The first device is the reference for the coordinate system

    # Dimensions
    pixel_size = [(250, 50)] * 4  # in um
    n_pixels = [(80, 336)] * 4
    z_positions = [0., 19500, 108800, 128300]  # in um
    dut_names = ("Tel_0", "Tel_1", "Tel_2", "Tel_3")

    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(os.path.split(data_files[0])[0], 'output_fei4')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # The following shows a complete test beam analysis by calling the seperate function in correct order

    # Cluster hits off all DUTs
    kwargs = [{  # Input parameters of the cluster function
        'input_hits_file': data_files[i],
        'max_x_distance': 2,
        'max_y_distance': 1,
        'max_time_distance': 2,
        'max_cluster_hits':1000,
        'dut_name': dut_names[i]} for i in range(0, len(data_files))]
    pool = Pool()
    for kwarg in kwargs:
        pool.apply_async(hit_analysis.cluster_hits, kwds=kwarg)  # Non blocking call of the cluster function, runs in seperate process
    pool.close()
    pool.join()

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(input_cluster_files=[data_file[:-3] + '_cluster.h5' for data_file in data_files],
                                    output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    dut_names=dut_names
                                    )

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=[data_file[:-3] + '_cluster.h5' for data_file in data_files],
                                     n_pixels=n_pixels,
                                     output_merged_file=os.path.join(output_folder, 'Merged.h5'),
                                     pixel_size=pixel_size)

    # Create prealignment data for the DUT positions to the first DUT from the correlations
    dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                               output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               z_positions=z_positions,
                               pixel_size=pixel_size,
                               s_n=0.1,
                               fit_background=False,
                               reduce_background=False,
                               dut_names=dut_names,
                               non_interactive=True)  # Tries to find cuts automatically; deactivate to do this manualy

    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
                                  input_alignment=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_aligned_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                                  force_prealignment=True)  # If there is already an alignment info in the alignment file this has to be set)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealigned.h5'))  # If there is already an alignment info in the alignment file this has to be set

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealigned.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                              fit_duts=[0, 1, 2, 3],
                              selection_track_quality=1,
                              force_prealignment=True)

    # Optional: plot some tracks (or track candidates) of a selected event range
    plot_utils.plot_events(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                           output_pdf=os.path.join(output_folder, 'Event.pdf'),
                           event_range=(0, 40),
                           dut=1)

    # Calculate the unconstrained residuals to check the alignment
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_residuals_file=os.path.join(output_folder, 'Residuals_prealigned.h5'),
                                        n_pixels=n_pixels,
                                        pixel_size=pixel_size,
                                        force_prealignment=True)

    # Calculate the efficiency and mean hit/track hit distance
    # When needed, set included column and row range for each DUT as list of tuples
    result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                         output_file=os.path.join(output_folder, 'Efficiency.h5'),
                                         output_pdf=os.path.join(output_folder, 'Efficiency.pdf'),
                                         bin_size=[(250, 50)],
                                         sensor_size=[(250. * 80, 50. * 336)],
                                         minimum_track_density=2,
                                         use_duts=None,
                                         cut_distance=500,
                                         max_distance=500,
                                         col_range=None,
                                         row_range=None,
                                         pixel_size=pixel_size,
                                         n_pixels=n_pixels,
                                         force_prealignment=True)

if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    run_analysis()