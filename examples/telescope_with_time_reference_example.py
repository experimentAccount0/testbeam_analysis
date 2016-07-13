'''Example script to run a full analysis on telescope data with device under tests and a time reference.

The telescope consists of 6 Mimosa26 planes and one FE-I4 with a full size planar n-in-n sensor as a timing reference.
The device under tests is a small passive CMOS sensor from LFoundry.

The Mimosa26 has an active area of 21.2mm x 10.6mm and the pixel matrix consists of 1152 columns and 576 rows (18.4um x 18.4um pixel size).
The total size of the chip is 21.5mm x 13.7mm x 0.036mm (radiation length 9.3660734)
The matrix is divided into 4 areas. For each area the threshold can be set up individually.
The quartes are from column 0-287, 288,575, 576-863 and 864-1151.
'''

import os
import logging
import numpy as np
from multiprocessing import Pool

import testbeam_analysis
from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows

    # Get the absolute example path
    #     tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(os.path.dirname(testbeam_analysis.__file__))) + r'/examples/data'))

    # The location of the example data files, one file per DUT
    data_files = [r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane0.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane1.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane2.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\fe_dut-converted-synchronized_plane0.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\fe_dut-converted-synchronized_plane1.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane3.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane4.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane5.h5']  # The first device is the reference for the coordinate system

    # Pixel dimesions and matrix size of the DUTs
    pixel_size = [(18.4, 18.4), (18.4, 18.4), (18.4, 18.4), (250, 50), (250, 50), (18.4, 18.4), (18.4, 18.4), (18.4, 18.4)]  # (Column, row) pixel pitch in um
    n_pixels = [(1152, 576), (1152, 576), (1152, 576), (80, 336), (80, 336), (1152, 576), (1152, 576), (1152, 576)]  # (Column, row) dimensions of the pixel matrix
    z_positions = [0., 20000, 40000, 40000 + 101000, 40000 + 101000 + 23000, 247000, 267000, 287000]  # in um
    dut_names = ("Tel 0", "Tel 1", "Tel 2", "LFCMOS3", "FEI4 Reference", "Tel 3", "Tel 4", "Tel 5")  # Friendly names for plotting

    # Folder where all output data and plots are stored
    output_folder = r'H:\Testbeam_05052016_LFCMOS\output'

    # The following shows a complete test beam analysis by calling the seperate function in correct order

# Remove hot pixel, only needed for devices wih noisy pixel like Mimosa 26
# A pool of workers to remove the noisy pixels in all files in parallel
#     threshold = (2, 2, 2, 10, 10, 2, 2, 2)
#     kwargs = [{
#         'input_hits_file': data_files[i],
#         'n_pixel': n_pixels[i],
#         'pixel_size': pixel_size[i],
#         'threshold': threshold[i],
#         'dut_name': dut_names[i]} for i in range(0, len(data_files))]
#     pool = Pool()
#     for kwarg in kwargs:
#         hit_analysis.remove_noisy_pixels(**kwarg)
#     pool.close()
#     pool.join()

# Cluster hits off all DUTs
# A pool of workers to cluster hits in all files in parallel
#     kwargs = [{
#         'input_hits_file': data_files[i][:-3] + '_noisy_pixels.h5',
#         'max_x_distance': 3,
#         'max_y_distance': 3,
#         'max_time_distance': 2,
#         'max_cluster_hits': 5000,
#         'dut_name': dut_names[i]} for i in range(0, len(data_files))]
#     pool = Pool()
#     for kwarg in kwargs:
#         pool.apply_async(hit_analysis.cluster_hits, kwds=kwarg)
#     pool.close()
#     pool.join()

# Correlate the row / column of each DUT
#     dut_alignment.correlate_cluster(input_cluster_files=[data_file[:-3] + '_noisy_pixels_cluster.h5' for data_file in data_files],
#                                     output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
#                                     n_pixels=n_pixels,
#                                     pixel_size=pixel_size,
#                                     dut_names=dut_names)

# Create prealignment relative to the first DUT from the correlation data
#     dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
#                                output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                z_positions=z_positions,
#                                pixel_size=pixel_size,
#                                dut_names=dut_names,
#                                fit_background=True,
# non_interactive=False)  # Tries to find cuts automatically; deactivate to do this manualy

# Merge the cluster tables to one merged table aligned at the event number
#     dut_alignment.merge_cluster_data(input_cluster_files=[data_file[:-3] + '_noisy_pixels_cluster.h5' for data_file in data_files],
#                                      output_merged_file=os.path.join(output_folder, 'Merged.h5'),
#                                      pixel_size=pixel_size)

# Apply the prealignment to the merged cluster table to create tracklets
#     dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
#                                   input_alignment=os.path.join(output_folder, 'Alignment.h5'),
#                                   output_hit_aligned_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
#                                   force_prealignment=True)

# Find tracks from the prealigned tracklets and stores the with quality indicator into track candidates table
#     track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
#                                input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'))

    # The following two steps are for demonstration only. They show track fitting and residual calculation on
    # prealigned hits. Usually you are not interessted in this and will use the aligned hits directly.

# # Step 1.: Fit the track candidates and create new track table (using the prealignment!)
#     track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
#                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                               output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
#                               selection_hit_duts=[0, 1, 2, 4, 5, 6, 7],
#                               selection_fit_duts=[0, 1, 2, 5, 6, 7],
#                               exclude_fit_dut=True,
#                               force_prealignment=True,  # This is just for demonstration purpose, you usually fully aligned hits
#                               track_quality=1)
# 
# # Step 2.:  Calculate the residuals to check the alignment (using the prealignment!)
#     result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
#                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                         output_residuals_file=os.path.join(output_folder, 'Residuals_prealigned.h5'),
#                                         n_pixels=n_pixels,
#                                         pixel_size=pixel_size,
#                                         use_duts=None,
#                                         force_prealignment=True,  # This is just for demonstration purpose, you usually use fully aligned hits
#                                         max_chi2=None)
#
# # Step 1.: Fit the track candidates and create new track table (using the prealignment!)
#     track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
#                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                               output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned_2_ref_next.h5'),
#                               fit_duts=[3],  # Fit tracks for all DUTs
#                               include_duts=[-1, 1, 2],  # Use only the DUT before and after the actual DUT for track fitting / interpolation
# #                               ignore_duts=[3],  # Ignore small LFCMOS sensor in track selection, to have better statisitcs for other devices
#                               ignore_fit_duts=[4],  # DUT4 is FE-I4 timing reference with hughe pixels, thus ignore it in fit but use it for selection
#                               force_prealignment=True,  # This is just for demonstration purpose, you usually fully aligned hits
#                               track_quality=1)

# # Step 2.:  Calculate the residuals to check the alignment (using the prealignment!)
#     result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned_2_ref_next.h5'),
#                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                         output_residuals_file=os.path.join(output_folder, 'Residuals_prealigned_2_ref_next.h5'),
#                                         n_pixels=n_pixels,
#                                         pixel_size=pixel_size,
#                                         use_duts=None,
#                                         force_prealignment=True,  # This is just for demonstration purpose, you usually use fully aligned hits
#                                         max_chi2=None)
    
# # Step 1.: Fit the track candidates and create new track table (using the prealignment!)
#     track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
#                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                               output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned_2_ref_outer.h5'),
# #                               fit_duts=[3],  # Fit tracks for all DUTs
#                               selection_hit_duts=[0, 1, 2, 5, 6, 7], 
#                               selection_fit_duts=[0, 1, 2, 5, 6, 7], 
#                               force_prealignment=True,  # This is just for demonstration purpose, you usually fully aligned hits
#                               track_quality=1)

# # Step 2.:  Calculate the residuals to check the alignment (using the prealignment!)
#     result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned_2_ref_outer.h5'),
#                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                         output_residuals_file=os.path.join(output_folder, 'Residuals_prealigned_2_ref_outer.h5'),
#                                         n_pixels=n_pixels,
#                                         pixel_size=pixel_size,
#                                         use_duts=None,
#                                         force_prealignment=True,  # This is just for demonstration purpose, you usually use fully aligned hits
#                                         max_chi2=None)

    from testbeam_analysis.tools import data_selection
    data_selection.select_hits(hit_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
                                   track_quality=0b11110111,
                                   track_quality_mask=0b11110111)

#     # Do an alignment step with the track candidates, corrects rotations and is therefore much more precise than simple prealignment
#     dut_alignment.alignment(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
#                             input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                             # align duts: order of planes to align, should start with high resoultion planes (here: telescope planes)
#                             align_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
#                                         [4],  # align the time reference after the telescope alignment
#                                         [3]],  # align the DUT last and not with tie reference since it is rather small
#                             selection_fit_duts=[[0, 1, 2, 5, 6, 7],
#                                                 [0, 1, 2, 5, 6, 7],
#                                                 [0, 1, 2, 5, 6, 7]],
#                             selection_hit_duts=[[0, 1, 2, 4, 5, 6, 7],
#                                                 [0, 1, 2, 4, 5, 6, 7],
#                                                 [0, 1, 2, 3, 4, 5, 6, 7]],
#                             selection_track_quality=[[1, 1, 1, 0, 1, 1, 1],
#                                                      [1, 1, 1, 1, 1, 1, 1],
#                                                      [1, 1, 1, 1, 0, 1, 1, 1]],
#                             initial_rotation=[[0., 0, 0.], 
#                                               [0., 0, 0.], 
#                                               [0, 0, 0.],
#                                               [np.pi - 0.05, -0.05, -0.005],
#                                               [np.pi - 0.01, -0.02, -0.0005], 
#                                               [0., 0, 0.], 
#                                               [0., 0, 0.], 
#                                               [0., 0, 0.]],
#                             initial_translation=[[0., 0, 0.], 
#                                               [0., 0, 0.],
#                                               [0., 0, 0.], 
#                                               [11540, 18791, 0.], 
#                                               [710., 9851., 0.], 
#                                               [0., 0, 0.], 
#                                               [0., 0, 0.], 
#                                               [0., 0, 0.]],
#                             n_pixels=n_pixels,
#                             use_n_tracks=200000,
#                             pixel_size=pixel_size)
#   
#     # Apply the alignment to the merged cluster table to create tracklets
#     dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
#                                   input_alignment=os.path.join(output_folder, 'Alignment.h5'),
#                                   output_hit_aligned_file=os.path.join(output_folder, 'Tracklets.h5'))
#     
#     # Find tracks from the tracklets and stores the with quality indicator into track candidates table
#     track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
#                                input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'))
#   
#     track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
#                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                               output_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
#                               fit_duts=[2],  # Fit tracks for all DUTs
#                               selection_hit_duts=[0, 1, 2, 5, 6, 7], 
#                               selection_fit_duts=[0, 1, 2, 5, 6, 7], 
#                               selection_track_quality=0)  # Take all tracks with hits, cut later on chi 2
#   
#     # Create unconstrained residuals
#     result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
#                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                         output_residuals_file=os.path.join(output_folder, 'Residuals.h5'),
#                                         n_pixels=n_pixels,
#                                         pixel_size=pixel_size)
