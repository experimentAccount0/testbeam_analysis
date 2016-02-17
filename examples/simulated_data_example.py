''' Example script to run a full analysis on simulated data.
'''

import logging
from multiprocessing import Pool

from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis import plot_utils

from testbeam_analysis.tools.simulate_data import SimulateData

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # main entry point is needed for multiprocessing under windows
    # Simulate 1000000 events with std. settings
    simulate_data = SimulateData(0)  # Start simulator with random seed = 0
    simulate_data.dut_efficiencies = [1., 0.99, 0.90, 0.8, 0.98, 1.]
    simulate_data.create_data_and_store('simulated_data', n_events=1000000)

    # The location of the data files, one file per DUT
    data_files = [r'simulated_data_DUT0.h5',  # the first DUT is the reference DUT defining the coordinate system, called internally DUT0
                  r'simulated_data_DUT1.h5',  # DUT1
                  r'simulated_data_DUT2.h5',  # DUT2
                  r'simulated_data_DUT3.h5',  # DUT3
                  r'simulated_data_DUT4.h5',  # DUT4
                  r'simulated_data_DUT5.h5'   # DUT5
                  ]

    # Dimensions
    pixel_size = [(50, 50), (50, 50), (50, 50), (50, 50), (50, 50), (50, 50)]  # in um
    n_pixel = [(1000, 1000), (1000, 1000), (1000, 1000), (1000, 1000), (1000, 1000), (1000, 1000)]
    z_positions = [0, 10000, 20000, 30000, 40000, 50000]  # in um

    output_folder = ''  # define a folder where all output data and plots are stored
    cluster_files = [data_file[:-3] + '_cluster.h5' for data_file in data_files]

    # The following shows a complete test beam analysis by calling the separate function in correct order

    # Cluster hits off all DUTs
    args = [{'data_file': data_files[i],
             'max_x_distance': 2,
             'max_y_distance': 1,
             'max_time_distance': 2,
             'max_cluster_hits':1000} for i in range(0, len(data_files))]
    pool = Pool()
    pool.map(hit_analysis.cluster_hits_wrapper, args)  # find cluster on all DUT data files in parallel on multiple cores
    pool.close()
    pool.join()
    plot_utils.plot_cluster_size(cluster_files,
                                 output_pdf='Cluster_Size.pdf')

    # Correlate the row / column of each DUT
    dut_alignment.correlate_hits(data_files,
                                 alignment_file='Correlation.h5')
    plot_utils.plot_correlations(alignment_file='Correlation.h5',
                                 output_pdf='Correlations.pdf')

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    # When needed, set offset and error cut for each DUT as list of tuples
    dut_alignment.coarse_alignment(correlation_file='Correlation.h5',
                                   alignment_file='Alignment.h5',
                                   output_pdf='Alignment.pdf',
                                   pixel_size=pixel_size)

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(cluster_files,
                                     alignment_file='Alignment.h5',
                                     tracklets_file='Tracklets.h5',
                                     pixel_size=pixel_size)

    dut_alignment.check_hit_alignment(tracklets_file='Tracklets.h5',
                                      output_pdf='Alignment_Check.pdf',
                                      combine_n_hits=1000000)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(tracklets_file='Tracklets.h5',
                               alignment_file='Alignment.h5',
                               track_candidates_file='TrackCandidates.h5')

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(track_candidates_file='TrackCandidates.h5',
                              tracks_file='Tracks.h5',
                              output_pdf='Tracks.pdf',
                              z_positions=z_positions,
                              include_duts=[-3, -2, -1, 1, 2, 3],
                              track_quality=1)

    # Optional: plot some tracks (or track candidates) of a selected event range
    plot_utils.plot_events(track_file='Tracks.h5',
                           output_pdf='Event.pdf',
                           z_positions=z_positions,
                           event_range=(0, 10),
                           dut=1)

    # Calculate the residuals to check the alignment
    result_analysis.calculate_residuals(tracks_file='Tracks.h5',
                                        output_pdf='Residuals.pdf',
                                        z_positions=z_positions)

    # Calculate the efficiency and mean hit/track hit distance
    # When needed, set included column and row range for each DUT as list of tuples
    result_analysis.calculate_efficiency(tracks_file='Tracks.h5',
                                         output_pdf='Efficiency.pdf',
                                         z_positions=z_positions,
                                         bin_size=(50, 50),
                                         minimum_track_density=2,
                                         use_duts=None,
                                         cut_distance=None,
                                         max_distance=500,
                                         col_range=None,
                                         row_range=None)
