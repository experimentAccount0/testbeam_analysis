'''Example script to run a full analysis on telescope data. The original data can be found in the example folder of the EuTelescope framework.
The telescope consists of 6 planes with 15 cm clearance between the planes. The residual for the second plane (DUT 1) is about 8 um and comparable
to the residuals from EuTelescope software (6 um).

The Mimosa26 has an active area of 21.2mm x 10.6mm and the pixel matrix consists of 1152 columns and 576 rows (18.4um x 18.4um pixel size).
The total size of the chip is 21.5mm x 13.7mm x 0.036mm (radiation length 9.3660734)

The matrix is divided into 4 areas. For each area the threshold can be set up individually.
The quartes are from column 0-287, 288,575, 576-863 and 864-1151.

The Mimosa26 detects ionizing particle with a density of up to 10^6 hits / cm^2 / s. The hit rate for a beam telescope is ~5 hits / frame.
'''

import os
import logging
from multiprocessing import Pool

import testbeam_analysis
from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows

    # Get the absolute example path
    tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(os.path.dirname(testbeam_analysis.__file__))) + r'/examples/data'))

    # The location of the example data files, one file per DUT
    data_files = [(os.path.join(tests_data_folder, r'TestBeamData_Mimosa26_DUT%d' % i + '.h5')) for i in range(6)]  # The first device is the reference for the coordinate system

    # Pixel dimesions and matrix size of the DUTs
    pixel_size = [(18.4, 18.4)] * 6  # Column, row pixel pitch in um
    n_pixels = [(1152, 576)] * 6  # Number of pixel on column, row

    z_positions = [0., 15000, 30000, 45000, 60000, 75000]  # z position in um
    dut_names = ("Tel_0", "Tel_1", "Tel_2", "Tel_3", "Tel_4", "Tel_5")  # Friednly names for plotting

    # Folder where all output data and plots are stored
    output_folder = os.path.split(data_files[0])[0]

    # The following shows a complete test beam analysis by calling the seperate function in correct order

    # Remove hot pixel, only needed for devices wih noisy pixel like Mimosa 26
    # A pool of workers to remove the noisy pixels in all files in parallel
    kwargs = [{
        'input_hits_file': data_files[i],
        'n_pixel': n_pixels[i],
        'pixel_size': pixel_size[i],
        'threshold': 5,
        'dut_name': dut_names[i]} for i in range(0, len(data_files))]
    pool = Pool()
    for kwarg in kwargs:
        pool.apply_async(hit_analysis.remove_noisy_pixels, kwds=kwarg)
    pool.close()
    pool.join()

    # Cluster hits off all DUTs
    # A pool of workers to cluster hits in all files in parallel
    kwargs = [{
        'input_hits_file': data_files[i][:-3] + '_noisy_pixels.h5',
        'max_x_distance': 3,
        'max_y_distance': 3,
        'max_time_distance': 2,
        'max_cluster_hits': 1000000,
        'dut_name': dut_names[i]} for i in range(0, len(data_files))]
    pool = Pool()
    for kwarg in kwargs:
        pool.apply_async(hit_analysis.cluster_hits, kwds=kwarg)
    pool.close()
    pool.join()

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(input_cluster_files=[data_file[:-3] + '_noisy_pixels_cluster.h5' for data_file in data_files],
                                    output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    dut_names=dut_names)

    # Create prealignment relative to the first DUT from the correlation data
    dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                               output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               z_positions=z_positions,
                               pixel_size=pixel_size,
                               dut_names=dut_names,
                               fit_background=True,  # This data has several tracks per event and noisy pixel, thus fit existing background
                               non_interactive=True)  # Tries to find cuts automatically; deactivate to do this manualy

    # Merge the cluster tables to one merged table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=[data_file[:-3] + '_noisy_pixels_cluster.h5' for data_file in data_files],
                                     output_merged_file=os.path.join(output_folder, 'Merged.h5'),
                                     pixel_size=pixel_size)

    # Apply the prealignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
                                  input_alignment=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_aligned_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                                  force_prealignment=True)

    # Find tracks from the prealigned tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'))

    # The following two steps are for demonstration only. They show track fitting and residual calculation on
    # prealigned hits. Usually you are not interessted in this and will use the aligned hits directly.

    # Step 1.: Fit the track candidates and create new track table (using the prealignment!)
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                              exclude_dut_hit=True,  # To get unconstrained residuals do not use DUT hit for track fit
                              force_prealignment=True,  # This is just for demonstration purpose, you usually fully aligned hits
                              selection_track_quality=1)

    # Step 2.:  Calculate the residuals to check the alignment (using the prealignment!)
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_residuals_file=os.path.join(output_folder, 'Residuals_prealigned.h5'),
                                        n_pixels=n_pixels,
                                        pixel_size=pixel_size,
                                        use_duts=None,
                                        force_prealignment=True,  # This is just for demonstration purpose, you usually use fully aligned hits
                                        max_chi2=None)

    # Do an alignment step with the track candidates, corrects rotations and is therefore much more precise than simple prealignment
    dut_alignment.alignment(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
                            input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                            n_pixels=n_pixels,
                            pixel_size=pixel_size,
                            align_duts=[0, 1, 2, 4, 5])

    # Apply the alignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
                                  input_alignment=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_aligned_file=os.path.join(output_folder, 'Tracklets.h5'))

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'))

    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                              exclude_dut_hit=True,  # To get unconstrained residuals do not use DUT hit for track fit
                              track_quality=1)

    # Create unconstrained residuals
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_residuals_file=os.path.join(output_folder, 'Residuals.h5'),
                                        n_pixels=n_pixels,
                                        pixel_size=pixel_size)
