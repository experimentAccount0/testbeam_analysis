''' Example script to run a full analysis on simulated data.
'''

import logging
import os
from multiprocessing import Pool

from testbeam_analysis import (hit_analysis, dut_alignment, track_analysis, result_analysis)

from testbeam_analysis.tools.simulate_data import SimulateData

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis(n_events):
    # Simulate 100000 events with std. settings
    simulate_data = SimulateData(random_seed=0)  # Start simulator with random seed 0

    # All simulator std. settings are listed here and can be changed
    # General setup
    simulate_data.n_duts = 6  # Number of DUTs in the simulation
    simulate_data.z_positions = [i * 10000 for i in range(simulate_data.n_duts)]  # in um; std: every 10 cm
    simulate_data.offsets = [(-10000 + 111 * 0., -10000 + 111 * 0.) for i in range(simulate_data.n_duts)]  # in x, y in mu
    simulate_data.rotations = [(0, 0, 0)] * simulate_data.n_duts  # in rotation around x, y, z axis in Rad
    simulate_data.temperature = 300  # Temperature in Kelvin, needed for charge sharing calculation
    # Beam related settings
    simulate_data.beam_position = (0, 0)  # Average beam position in x, y at z = 0 in mu
    simulate_data.beam_position_sigma = (2000, 2000)  # in x, y at z = 0 in mu
    simulate_data.beam_momentum = 3200  # Beam momentum in MeV
    simulate_data.beam_angle = 0  # Average beam angle in theta at z = 0 in mRad
    simulate_data.beam_angle_sigma = 2  # Deviation from the average beam angle in theta at z = 0 in mRad
    simulate_data.tracks_per_event = 3  # Average number of tracks per event
    simulate_data.tracks_per_event_sigma = 1  # Deviation from the average number of tracks, makes no track per event possible!
    # Device specific settings
    simulate_data.dut_bias = [50] * simulate_data.n_duts  # Sensor bias voltage for each device in volt
    simulate_data.dut_thickness = [200] * simulate_data.n_duts  # Sensor thickness for each device in um
    simulate_data.dut_threshold = [0.] * simulate_data.n_duts  # Detection threshold for each device in electrons, influences efficiency!
    simulate_data.dut_noise = [0.] * simulate_data.n_duts  # Noise for each device in electrons
    simulate_data.dut_pixel_size = [(50, 18.4)] * simulate_data.n_duts  # Pixel size for each device in x / y in um
    simulate_data.dut_n_pixel = [(400, 1100)] * simulate_data.n_duts  # Number of pixel for each device in x / y
    simulate_data.dut_efficiencies = [1.] * simulate_data.n_duts  # Efficiency for each device from 0. to 1. for hits above threshold
    simulate_data.dut_material_budget = [simulate_data.dut_thickness[i] * 1e-4 / 9.370 for i in range(simulate_data.n_duts)]  # The effective material budget (sensor + passive compoonents) given in total material distance / total radiation length (https://cdsweb.cern.ch/record/1279627/files/PH-EP-Tech-Note-2010-013.pdf); 0 means no multiple scattering; std. setting is the sensor thickness made of silicon as material budget
    # Digitization settings
    simulate_data.digitization_charge_sharing = True
    simulate_data.digitization_shuffle_hits = True  # Shuffle hit per event to challenge track finding
    simulate_data.digitization_pixel_discretization = True  # Translate hit position on DUT plane to channel indices (column / row)

    # Create the data
    output_folder = 'simulation'  # Define a folder where all output data and plots are stored
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    simulate_data.create_data_and_store(os.path.join(output_folder, 'simulated_data'), n_events=n_events)

    # The simulated data files, one file per DUT
    data_files = [os.path.join(output_folder, r'simulated_data_DUT%d.h5' % i) for i in range(simulate_data.n_duts)]

    # The following shows a complete test beam analysis by calling the separate function in correct order

    # Cluster hits off all DUTs
    kwargs = [{
        'input_hits_file': data_files[i],
        'max_x_distance': 1,
        'max_y_distance': 1,
        'max_time_distance': 2,
        'max_hit_charge': 2 ** 16,
        "dut_name": data_files[i]} for i in range(len(data_files))]
    pool = Pool()
    for kwarg in kwargs:
        pool.apply_async(hit_analysis.cluster_hits, kwds=kwarg)
    pool.close()
    pool.join()

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(input_cluster_files=[data_file[:-3] + '_cluster.h5' for data_file in data_files],
                                    output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                    n_pixels=simulate_data.dut_n_pixel,
                                    pixel_size=simulate_data.dut_pixel_size)

    # Create alignment data for the DUT positions to the first DUT from the correlation data
    # When needed, set offset and error cut for each DUT as list of tuples
    dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                               output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               z_positions=simulate_data.z_positions,
                               pixel_size=simulate_data.dut_pixel_size,
                               no_fit=True,  # Deactivate if you have a large dataset, enhances alignment slightly
                               fit_background=False if not (simulate_data.tracks_per_event or
                                                            simulate_data.tracks_per_event_sigma) else True,
                               non_interactive=True)  # Tries to find cuts automatically; deactivate to do this manualy

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=[data_file[:-3] + '_cluster.h5' for data_file in data_files],
                                     output_merged_file=os.path.join(output_folder, 'Merged.h5'),
                                     n_pixels=simulate_data.dut_n_pixel,
                                     pixel_size=simulate_data.dut_pixel_size)

    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
                                  input_alignment=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_aligned_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                                  force_prealignment=True)  # If there is already an alignment info in the alignment file this has to be set)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealigned.h5'),
                               min_cluster_distance=False)  # If there is already an alignment info in the alignment file this has to be set)

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealigned.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
                              exclude_dut_hit=True,  # To get unconstrained residuals do not use DUT hit for track fit
                              selection_track_quality=0,
                              min_track_distance=1000,  # To get close to excact efficiency heavily avoid merged tracks
                              force_prealignment=True)

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
                                         pixel_size=simulate_data.dut_pixel_size,
                                         n_pixels=simulate_data.dut_n_pixel,
                                         force_prealignment=True,
                                         show_inefficient_events=True)

if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    run_analysis(n_events=1000000)
