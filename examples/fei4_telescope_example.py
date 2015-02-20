''' Example script to run a full analysis on FE-I4 telescope data. The original data was recorded at DESY with pyBar. 
The telescope consists of 6 DUTs with ~ 2 cm distance between the planes. Onle the first two and last two planes were taken here. 
The first and last plane were IBL n-in-n planar sensors and the 2 devices in the middle 3D CNM/FBK sensors.
'''

import os
from multiprocessing import Pool
import pyTestbeamAnalysis.analyze_test_beam as atb

if __name__ == '__main__':
    # The location of the datafiles, one file per DUT
    data_files = ['data\\TestBeamData_FEI4_DUT0.h5',  # the first DUT is the reference DUT defining the coordinate system
                  'data\\TestBeamData_FEI4_DUT1.h5',
                  'data\\TestBeamData_FEI4_DUT4.h5',
                  'data\\TestBeamData_FEI4_DUT5.h5'
                  ]

    # Dimesions
    pixel_size = (250, 50)  # um
    z_positions = [0., 1.95, 10.88, 12.83]  # in cm; optional, can be also deduced from data, but usually not with high precision (~ mm)

    output_folder = os.path.split(data_files[0])[0]  # define a folder where all output data and plots are stored
    cluster_files = [data_file[:-3] + '_cluster.h5' for data_file in data_files]

    # The following shows a complete test beam analysis by calling the seperate function in correct order

    # Correlate the row/col of each DUT
    atb.correlate_hits(data_files, alignment_file=output_folder + '\\Alignment.h5', fraction=1)
    atb.plot_correlations(alignment_file=output_folder + '\\Alignment.h5', output_pdf=output_folder + '\\Correlations.pdf')

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    atb.align_hits(alignment_file=output_folder + '\\Alignment.h5', combine_bins=1, no_data_cut=0.7, fit_error_cut=(2.0, 2.0), output_pdf=output_folder + '\\Alignment.pdf')

    # Cluster hits off all DUTs
    Pool().map(atb.cluster_hits, data_files)  # find cluster on all DUT data files in parallel on multiple cores
    atb.plot_cluster_size(cluster_files, output_pdf=output_folder + '\\Cluster_Size.pdf')

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    atb.merge_cluster_data(cluster_files, alignment_file=output_folder + '\\Alignment.h5', tracklets_file=output_folder + '\\Tracklets.h5')

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    atb.find_tracks(tracklets_file=output_folder + '\\Tracklets.h5', alignment_file=output_folder + '\\Alignment.h5', track_candidates_file=output_folder + '\\TrackCandidates.h5')

    # optional: try to deduce the devices z positions. Difficult for parallel tracks / bad resolution and does actually not really help here. Still good for cross check.
    atb.align_z(track_candidates_file=output_folder + '\\TrackCandidates.h5', alignment_file=output_folder + '\\Alignment.h5', output_pdf=output_folder + '\\Z_positions.pdf', z_positions=z_positions, track_quality=2, max_tracks=1, warn_at=0.5)

    # Fit the track candidates and create new track table
    atb.fit_tracks(track_candidates_file=output_folder + '\\TrackCandidates.h5', tracks_file=output_folder + '\\Tracks.h5', output_pdf=output_folder + '\\Tracks.pdf', z_positions=z_positions, fit_duts=None, include_duts=[-3, -2, -1, 1, 2, 3], ignore_duts=None, max_tracks=1, track_quality=1, pixel_size=pixel_size)

    # optional: plot some tracks (or track candidates) of a selected event ragnge
    atb.event_display(track_file=output_folder + '\\Tracks.h5', output_pdf=output_folder + '\\Event.pdf', z_positions=z_positions, event_range=(0, 10), pixel_size=pixel_size, plot_lim=(2, 2), dut=1)

    # Calculate the residuals to check the alignment
    atb.calculate_residuals(tracks_file=output_folder + '\\Tracks.h5', output_pdf=output_folder + '\\Residuals.pdf', z_positions=z_positions, pixel_size=pixel_size, use_duts=None, track_quality=2, max_chi2=3e3)

    # Plot the track density on selected DUT planes
    atb.plot_track_density(tracks_file=output_folder + '\\Tracks.h5', output_pdf=output_folder + '\\TrackDensity.pdf', z_positions=z_positions, use_duts=None)

    # Calculate the efficiency and mean hit/track hit distance
    atb.calculate_efficiency(tracks_file=output_folder + '\\Tracks.h5', output_pdf=output_folder + '\\Efficiency.pdf', z_positions=z_positions, minimum_track_density=2, pixel_size=pixel_size, use_duts=None)
