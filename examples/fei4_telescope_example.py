''' Example script to run a full analysis on FE-I4 telescope data. The original data was recorded at DESY with pyBar.
The telescope consists of 6 DUTs with ~ 2 cm distance between the planes. Only the first two and last two planes were taken here.
The first and last plane were IBL n-in-n planar sensors and the 2 devices in the middle 3D CNM/FBK sensors.
'''

import os
import logging
from multiprocessing import Pool

from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis import plot_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # main entry point is needed for multiprocessing under windows
    # The location of the data files, one file per DUT
    data_files = [r'data/TestBeamData_FEI4_DUT0.h5',  # the first DUT is the reference DUT defining the coordinate system, called internally DUT0
                  r'data/TestBeamData_FEI4_DUT1.h5',  # DUT1
                  r'data/TestBeamData_FEI4_DUT4.h5',  # DUT2
                  r'data/TestBeamData_FEI4_DUT5.h5'   # DUT3
                  ]

    # Dimensions
    pixel_size = [(250, 50)] * 4  # in um
    n_pixels = [(80, 336)] * 4
    z_positions = [0., 19500, 108800, 128300]  # in um; optional, can be also deduced from data, but usually not with high precision (~ mm)
    dut_names = ("Tel_0", "Tel_1", "Tel_2", "Tel_3")

    output_folder = os.path.split(data_files[0])[0]  # define a folder where all output data and plots are stored

    # The following shows a complete test beam analysis by calling the seperate function in correct order

# FIXME: need major rework
# Create the initial geometry (to be done once)
#     geometry_utils.create_initial_geometry(geo_file, z_positions)
#     geo_file = os.path.join(output_folder, 'FEI4Geometry.h5')

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
    # free resources
    pool.close()
    pool.join()
    cluster_files = [res.get() for res in multiple_results]

    # Correlate the row / column of each DUT
    dut_alignment.correlate_hits(input_hits_files=data_files,
                                 output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                 n_pixels=n_pixels,
                                 pixel_size=pixel_size,
                                 dut_names=dut_names
                                 )

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    dut_alignment.coarse_alignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                   output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                   output_pdf_file=os.path.join(output_folder, 'Alignment.pdf'),
                                   pixel_size=pixel_size,
                                   non_interactive=True)  # Tries to find cuts automatically; deactivate to do this manualy

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=cluster_files,
                                     input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                     output_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                                     pixel_size=pixel_size)

    dut_alignment.check_hit_alignment(input_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                                      output_pdf_file=os.path.join(output_folder, 'Alignment_Check.pdf'),
                                      combine_n_hits=1000000)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'))

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                              output_pdf_file=os.path.join(output_folder, 'Tracks.pdf'),
                              z_positions=z_positions,
                              fit_duts=[0, 1, 2, 3],
                              include_duts=[-3, -2, -1, 1, 2, 3],
                              track_quality=1)

    # Optional: plot some tracks (or track candidates) of a selected event range
    plot_utils.plot_events(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                           output_pdf=os.path.join(output_folder, 'Event.pdf'),
                           z_positions=z_positions,
                           event_range=(0, 10),
                           dut=1)

    # Calculate the residuals to check the alignment
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                        output_pdf=os.path.join(output_folder, 'Residuals.pdf'),
                                        z_positions=z_positions,
                                        max_chi2=10000)

    # Plot the track density on selected DUT planes
    plot_utils.plot_track_density(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                  output_pdf=os.path.join(output_folder, 'TrackDensity.pdf'),
                                  z_positions=z_positions,
                                  dim_x=80,
                                  dim_y=336,
                                  pixel_size=pixel_size,
                                  use_duts=None)

    plot_utils.plot_charge_distribution(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                                        output_pdf=os.path.join(output_folder, 'ChargeDistribution.pdf'),
                                        dim_x=(80, 80, 80, 80),
                                        dim_y=(336, 336, 336, 336),
                                        pixel_size=pixel_size)

    # Calculate the efficiency and mean hit/track hit distance
    # When needed, set included column and row range for each DUT as list of tuples
    result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                         output_pdf=os.path.join(output_folder, 'Efficiency.pdf'),
                                         z_positions=z_positions,
                                         bin_size=(250, 50),
                                         minimum_track_density=2,
                                         use_duts=None,
                                         cut_distance=500,
                                         max_distance=500,
                                         col_range=(1250, 17500),
                                         row_range=(1000, 16000))
