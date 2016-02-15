''' Example script to run a full analysis on telescope data. The original data can be found in the example folder of the EuTelescope framework. 
The telescope consists of 6 planes with 15 cm distance between the planes. The residuals for the second plane (DUT 1) are about 8 um and comparable 
to the residuals from EuTelescope (6 um).

The other plane residuals are not that small depicting a worse performance in device algnment and track fitting. 
'''

import os
import logging
from multiprocessing import Pool

from testbeam_analysis import hit_analysis, geometry_utils
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis import plot_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # main entry point is needed for multiprocessing under windows
    # The location of the data files, one file per DUT
    data_files = [r'data/TestBeamData_Mimosa26_DUT0.h5',  # The first DUT (DUT 0) is the reference DUT defining the coordinate system
                  r'data/TestBeamData_Mimosa26_DUT1.h5',  # Data file of DUT 1
                  r'data/TestBeamData_Mimosa26_DUT2.h5',  # Data file of DUT 2
                  r'data/TestBeamData_Mimosa26_DUT3.h5',  # Data file of DUT 3
                  r'data/TestBeamData_Mimosa26_DUT4.h5',  # Data file of DUT 4
                  r'data/TestBeamData_Mimosa26_DUT5.h5',  # Data file of DUT 5
                  ]

    # Pixel dimesions and matrix size of the DUTs
    pixel_size = [(18.4, 18.4),  # Column, row pixel pitch in um of DUT 0
                  (18.4, 18.4),  # Column, row pixel pitch in um of DUT 1
                  (18.4, 18.4),  # Column, row pixel pitch in um of DUT 2
                  (18.4, 18.4),  # Column, row pixel pitch in um of DUT 3
                  (18.4, 18.4),  # Column, row pixel pitch in um of DUT 4
                  (18.4, 18.4)]  # Column, row pixel pitch in um of DUT 5
    n_pixels = [(1152, 576),  # Number of pixels on column, row for DUT 0
                (1152, 576),  # Number of pixels on column, row for DUT 1
                (1152, 576),  # Number of pixels on column, row for DUT 2
                (1152, 576),  # Number of pixels on column, row for DUT 3
                (1152, 576),  # Number of pixels on column, row for DUT 4
                (1152, 576)]  # Number of pixels on column, row for DUT 5

    z_positions = [0., 15000, 30000, 45000, 60000, 75000]  # in um optional, can be also deduced from data, but usually not with high precision (~ mm)

    output_folder = os.path.split(data_files[0])[0]  # define a folder where all output data and plots are stored
    geoFile = r'data/MimosaGeometry.h5'
    
    # The following shows a complete test beam analysis by calling the seperate function in correct order

    #Create the initial geometry (to be done once)
    geometry_utils.create_initial_geometry (geoFile, z_positions)
                            
    geometry_utils.update_rotation_angle(geoFile, 0, -0.00104)
    geometry_utils.update_rotation_angle(geoFile, 1, 0.00104)
    geometry_utils.update_rotation_angle(geoFile, 2, 0.00102)
    geometry_utils.update_translation_val(geoFile, 1, -10.7, 19.4)
    geometry_utils.update_translation_val(geoFile, 2, -12.1, 17.4)

    # Remove hot pixels, only needed for devices wih noisy pixels like Mimosa 26
    Pool().map(hit_analysis.remove_noisy_pixels, data_files)  # delete noisy hits in DUT data files in parallel on multiple cores
    data_files = [data_file[:-3] + '_hot_pixel.h5' for data_file in data_files]
    cluster_files = [data_file[:-3] + '_cluster.h5' for data_file in data_files]

    # Cluster hits off all DUTs
    args = [(data_files[i], 3, 3, 2, 1000000) for i in range(0, len(data_files))]
    Pool().map(hit_analysis.cluster_hits_wrapper, args)  # find cluster on all DUT data files in parallel on multiple cores
    plot_utils.plot_cluster_size(cluster_files,
                                 output_pdf=output_folder + r'/Cluster_Size.pdf')

    # Correlate the row / column of each DUT
    dut_alignment.correlate_hits(data_files,
                                 alignment_file=output_folder + r'/Correlation.h5', fraction=1)
    plot_utils.plot_correlations(alignment_file=output_folder + r'/Correlation.h5',
                                 output_pdf=output_folder + r'/Correlations.pdf')

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    # When needed, set offset and error cut for each DUT as list of tuples
    dut_alignment.align_hits(correlation_file=output_folder + r'/Correlation.h5',
                             alignment_file=output_folder + r'/Alignment.h5',
                             output_pdf=output_folder + r'/Alignment.pdf',
                             fit_offset_cut=[(800. / 10., 200. / 10.),  # Offset cut for DUT0 in columns, row
                                             (1000. / 10., 1000. / 10.),  # Offset cut for DUT1 in columns, row
                                             (1200. / 10., 1200. / 10.),
                                             (1400. / 10., 1400. / 10.),
                                             (1800. / 10., 1800. / 10.)],  # Offset cut for all DUT rows
                             fit_error_cut=[(4000. / 1000., 2200. / 1000.),  # Fit error cut for DUT1 in columns, row
                                            (9000. / 1000., 9000. / 1000.),  # Fit error cut for DUT2 in columns, row
                                            (20000. / 1000., 20000. / 1000.),
                                            (25000. / 1000., 25000. / 1000.),
                                            (35000. / 1000., 35000. / 1000.)],
                             pixel_size=pixel_size,
                             show_plots=False)

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(cluster_files,
                                     alignment_file=output_folder + r'/Alignment.h5',
                                     tracklets_file=output_folder + r'/Tracklets.h5',
                                     pixel_size=pixel_size)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(tracklets_file=output_folder + r'/Tracklets.h5',
                               alignment_file=output_folder + r'/Alignment.h5',
                               track_candidates_file=output_folder + r'/TrackCandidates.h5')

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(track_candidates_file=output_folder + r'/TrackCandidates.h5',
                              tracks_file=output_folder + r'/Tracks.h5',
                              output_pdf=output_folder + r'/Tracks.pdf',
                              z_positions=z_positions,
                              fit_duts=[1, 2, 3, 4],  # Fit tracks for all DUTs
                              include_duts=[-1, 1],  # Use only the DUT before and after the actual DUT for track fitting / interpolation
                              ignore_duts=None,
                              track_quality=2,
                              method='Interpolation',
                              pixel_size=pixel_size,
                              geometryFile=geoFile)

    # Calculate the residuals to check the alignment
    result_analysis.calculate_residuals(tracks_file=output_folder + r'/Tracks.h5',
                                        output_pdf=output_folder + r'/Residuals.pdf',
                                        z_positions=z_positions,
                                        use_duts=None,
                                        max_chi2=None,
                                        method='Interpolation',
                                        geometryFile=geoFile)
