''' Example script to run a full analysis on telescope data. The original data can be found in the example folder of the EuTelescope framework. 
The telescope consists of 6 planes with 15 cm distance between the planes. Onle the first three planes were taken here, thus the line fit chi2 
is alays 0. The residuals for the second plane (DUT 1) are about 8 um and comparable to the residuals from EuTelescope (6 um).
'''

import os
from multiprocessing import Pool
import pyTestbeamAnalysis.analyze_test_beam as atb

if __name__ == '__main__':
    # The location of the datafiles, one file per DUT
    data_files = [r'data/TestBeamData_Mimosa26_DUT0.h5',  # the first DUT is the reference DUT defining the coordinate system
                  r'data/TestBeamData_Mimosa26_DUT1.h5',
                  r'data/TestBeamData_Mimosa26_DUT2.h5',
                  ]

    # Dimesions
    pixel_size = (18.4, 18.4)  # um
    z_positions = [0., 15., 30., 45., 60., 75.]  # in cm; optional, can be also deduced from data, but usually not with high precision (~ mm)

    output_folder = os.path.split(data_files[0])[0]  # define a folder where all output data and plots are stored

    # The following shows a complete test beam analysis by calling the seperate function in correct order

    # Remove hot pixels, only needed for devices wih noisy pixels like Mimosa 26
    Pool().map(atb.remove_hot_pixels, data_files)  # delete noisy hits in DUT data files in parallel on multiple cores
    data_files = [data_file[:-3] + '_hot_pixel.h5' for data_file in data_files]
    cluster_files = [data_file[:-3] + '_cluster.h5' for data_file in data_files]

    # Correlate the row/col of each DUT
    atb.correlate_hits(data_files, alignment_file=output_folder + r'/Alignment.h5', fraction=1)
    atb.plot_correlations(alignment_file=output_folder + r'/Alignment.h5', output_pdf=output_folder + r'/Correlations.pdf')

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    atb.align_hits(alignment_file=output_folder + r'/Alignment.h5', output_pdf=output_folder + r'/Alignment.pdf', fit_offset_cut=(40. / 10., 10. / 10.), fit_error_cut=(500. / 1000., 500. / 1000.))

    # Cluster hits off all DUTs
    Pool().map(atb.cluster_hits, data_files)  # find cluster on all DUT data files in parallel on multiple cores
    atb.plot_cluster_size(cluster_files, output_pdf=output_folder + r'/Cluster_Size.pdf')

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    atb.merge_cluster_data(cluster_files, alignment_file=output_folder + r'/Alignment.h5', tracklets_file=output_folder + r'/Tracklets.h5')

    # Check alignment of hits in position and time
    atb.check_hit_alignment(output_folder + r'/Tracklets.h5', output_folder + r'/Alignment_Check.pdf')

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    atb.find_tracks(tracklets_file=output_folder + r'/Tracklets.h5', alignment_file=output_folder + r'/Alignment.h5', track_candidates_file=output_folder + r'/TrackCandidates.h5')

    # optional: try to deduce the devices z positions. Difficult for parallel tracks / bad resolution and does actually not really help here. Still good for cross check.
    atb.align_z(track_candidates_file=output_folder + r'/TrackCandidates.h5', alignment_file=output_folder + r'/Alignment.h5', output_pdf=output_folder + r'/Z_positions.pdf', z_positions=z_positions, track_quality=2, max_tracks=1, warn_at=0.5)

    # Fit the track candidates and create new track table
    atb.fit_tracks(track_candidates_file=output_folder + r'/TrackCandidates.h5', tracks_file=output_folder + r'/Tracks.h5', output_pdf=output_folder + r'/Tracks.pdf', z_positions=z_positions, fit_duts=[0, 1, 2], include_duts=[-2, -1, 1, 2], ignore_duts=None, max_tracks=4, track_quality=2, pixel_size=pixel_size)
#
# optional: plot some tracks (or track candidates) of a selected event ragnge
#     atb.event_display(track_file=output_folder + r'/Tracks.h5', output_pdf=output_folder + r'/Event.pdf', z_positions=z_positions, event_range=(6493424, 6493425), pixel_size=pixel_size, plot_lim=(2, 2), dut=1)

    # Calculate the residuals to check the alignment
    atb.calculate_residuals(tracks_file=output_folder + r'/Tracks.h5', output_pdf=output_folder + r'/Residuals.pdf', z_positions=z_positions, pixel_size=pixel_size, use_duts=None, track_quality=2, max_chi2=3e3)
