''' Example script to run a full analysis on FE-I4 telescope data. The original data was recorded at DESY with pyBar.
The telescope consists of 6 DUTs with ~ 2 cm distance between the planes. Only the first two and last two planes were taken here.
The first and last plane were IBL n-in-n planar sensors and the 2 devices in the middle 3D CNM/FBK sensors.
'''

import os
import logging
from multiprocessing import Pool

import testbeam_analysis
from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis import plot_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # main entry point is needed for multiprocessing under windows
    # Get the absolute example path, only needed to test this example
    tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(os.path.dirname(testbeam_analysis.__file__))) + r'/examples/data'))
    # The location of the data files, one file per DUT
    data_files = [(os.path.join(tests_data_folder, r'TestBeamData_FEI4_DUT%d' % i + '.h5')) for i in [0, 1, 4, 5]]  # The first device is the reference for the coordinate system

    # Dimensions
    pixel_size = [(250, 50)] * 4  # in um
    n_pixels = [(80, 336)] * 4
    z_positions = [0., 19500, 108800, 128300]  # in um; optional, can be also deduced from data, but usually not with high precision (~ mm)
    dut_names = ("Tel_0", "Tel_1", "Tel_2", "Tel_3")

    output_folder = os.path.split(data_files[0])[0]  # define a folder where all output data and plots are stored

    # The following shows a complete test beam analysis by calling the seperate function in correct order

    # Cluster hits off all DUTs
    kwargs = [{
        'input_hits_file': data_files[i],
        'max_x_distance': 2,
        'max_y_distance': 1,
        'max_time_distance': 2,
        'max_cluster_hits':1000,
        'dut_name': dut_names[i]} for i in range(0, len(data_files))]
    pool = Pool()
    multiple_results = [pool.apply_async(hit_analysis.cluster_hits, kwds=kwarg) for kwarg in kwargs]
    pool.close()
    pool.join()

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(input_cluster_files=[data_file[:-3] + '_cluster.h5' for data_file in data_files],
                                    output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    dut_names=dut_names
                                    )

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                               output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               z_positions=z_positions,
                               pixel_size=pixel_size,
                               dut_names=dut_names,
                               non_interactive=True)  # Tries to find cuts automatically; deactivate to do this manualy

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=[data_file[:-3] + '_cluster.h5' for data_file in data_files],
                                     output_merged_file=os.path.join(output_folder, 'Merged.h5'),
                                     pixel_size=pixel_size)

    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
                                  input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_aligned_file=os.path.join(output_folder, 'Tracklets.h5'))

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'))

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                              output_pdf_file=os.path.join(output_folder, 'Tracks.pdf'),
                              fit_duts=[0, 1, 2, 3],
                              include_duts=[-3, -2, -1, 1, 2, 3],
                              track_quality=2)

    # Optional: plot some tracks (or track candidates) of a selected event range
    plot_utils.plot_events(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                           output_pdf=None,  # os.path.join(output_folder, 'Event.pdf'),
                           event_range=(0, 40),
                           dut=1)

    # Calculate the residuals to check the alignment
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_pdf=os.path.join(output_folder, 'Residuals.pdf'),
                                        max_chi2=10000)

    # Calculate the efficiency and mean hit/track hit distance
    # When needed, set included column and row range for each DUT as list of tuples
    result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                         output_pdf=os.path.join(output_folder, 'Efficiency.pdf'),
                                         bin_size=(250, 50),
                                         minimum_track_density=2,
                                         use_duts=None,
                                         cut_distance=500,
                                         max_distance=500,
                                         col_range=None,
                                         row_range=None)
