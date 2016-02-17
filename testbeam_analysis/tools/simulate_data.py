''' Helper functions for the unittests are defined here.
'''

import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import logging
import progressbar

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


class SimulateData(object):

    def __init__(self, random_seed=None):
        np.random.seed(random_seed)  # Set the random number seed to be able to rerun with same data
        self.reset()

    def set_std_settings(self):
        # Setup settings
        self.n_duts = 6
        self.z_positions = [i * 10000 for i in range(self.n_duts)]  # in um; st: every 10 cm
        self.offsets = [(-2500, -2500)] * self.n_duts  # in x, y in mu

        # Beam settings
        self.beam_position = (0, 0)  # in x, y at z = 0 in mu
        self.beam_position_sigma = (2000, 2000)  # in x, y at z = 0 in mu
        self.beam_angle = 0  # in theta at z = 0 in mRad
        self.beam_angle_sigma = 1  # in theta at z = 0 in mRad
        self.tracks_per_event = 1
        self.tracks_per_event_sigma = 1

        # Device settings
        self.dut_cluster_size = 1
        self.dut_cluster_size_sigma = 1
        self.dut_pixel_size = [(50, 50)] * self.n_duts  # pixel size in x / y in um
        self.dut_n_pixel = [(1000, 1000)] * self.n_duts  # number of pixelk in x / y
        self.dut_efficiencies = [1.] * self.n_duts  # dimension in x / y in um

        # Internals
        self._hit_dtype = np.dtype([('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])

    def reset(self):
        self.set_std_settings()
        self._hit_files = None

    def get_hit_files(self):
        return self._hit_files

    def _create_tracks(self, n_tracks):
        '''Creates tracks with gaussian distributed angles at gaussian distributed positions at z=0.

        Parameters
        ----------
        n_tracks: number
            Number of tracks created

        Returns
        -------
        Four np.arrays with position x,y and angles phi, theta
        '''

        logging.debug('Create %d tracks at x/y = (%d/%d +- %d/%d) um and theta = (%d +- %d) mRad', n_tracks, self.beam_position[0], self.beam_position[1], self.beam_position_sigma[0], self.beam_position_sigma[1], self.beam_angle, self.beam_angle_sigma)

        if self.beam_angle / 1000. > np.pi or self.beam_angle / 1000. < 0:
            raise ValueError('beam_angle has to be between [0..pi] Rad')

        if self.beam_position_sigma[0] != 0:
            track_positions_x = np.random.normal(self.beam_position[0], self.beam_position_sigma[0], n_tracks)

        else:
            track_positions_x = np.repeat(self.beam_position[0], repeats=n_tracks)  # Constant x = mean_x

        if self.beam_position_sigma[1] != 0:
            track_positions_y = np.random.normal(self.beam_position[1], self.beam_position_sigma[0], n_tracks)

        else:
            track_positions_y = np.repeat(self.beam_position[1], repeats=n_tracks)  # Constant y = mean_y

        if self.beam_angle_sigma != 0:
            track_angles_theta = np.abs(np.random.normal(self.beam_angle / 1000., self.beam_angle_sigma / 1000., size=n_tracks))  # Gaussian distributed theta
        else:  # Allow sigma = 0
            track_angles_theta = np.repeat(self.beam_angle / 1000., repeats=n_tracks)  # Constant theta = 0

        # Cut down to theta = 0 .. Pi
        while(np.any(track_angles_theta > np.pi) or np.any(track_angles_theta < 0)):
            track_angles_theta[track_angles_theta > np.pi] = np.random.normal(self.beam_angle, self.beam_angle_sigma, size=track_angles_theta[track_angles_theta > np.pi].shape[0])
            track_angles_theta[track_angles_theta < 0] = np.random.normal(self.beam_angle, self.beam_angle_sigma, size=track_angles_theta[track_angles_theta < 0].shape[0])

        track_angles_phi = np.random.random(size=n_tracks) * 2 * np.pi  # Flat distributed phi = [0, Pi[

        return track_positions_x, track_positions_y, track_angles_phi, track_angles_theta

    def _create_hits_from_tracks(self, track_positions_x, track_positions_y, track_angles_phi, track_angles_theta):
        '''Creates exact intersection points (x, y) at the given DUT z_positions for the given tracks. The tracks are defined with with the position at z = 0 (track_positions_x, track_positions_y) and
        an angle (track_angles_phi, track_angles_theta).

        Returns
        -------
        Two np.arrays with position, angle
        '''
        logging.debug('Intersect tracks with DUTs to create hits')

        intersections = []
        track_positions = np.column_stack((track_positions_x, track_positions_y))  # Track position at z = 0
        for z_position in self.z_positions:
            r = z_position / np.cos(track_angles_theta)  # r in spherical coordinates at actual z_position
            extrapolate = (r * np.array([np.cos(track_angles_phi) * np.sin(track_angles_theta), np.sin(track_angles_phi) * np.sin(track_angles_theta)])).T
            intersections.append(track_positions + extrapolate)
        return intersections

    def _digitize_hits(self, hits):
        ''' Takes the Monte Carlo hits and transfers them to the local DUT coordinate system and discretizes the position'''
        logging.debug('Digitize hits')
        digitized_hits = []
        for dut_index, dut_hits in enumerate(hits):
            dut_hits -= np.array(self.offsets[dut_index])  # Add DUT offset
            dut_hits /= np.array(self.dut_pixel_size[dut_index])  # Position in pixel numbers
            dut_hits = dut_hits.astype(np.int32) + 1  # Pixel discretization, column/row index start from 1

            # Mask hit outside of the DUT
            selection_x = np.logical_or(dut_hits.T[0] < 1, dut_hits.T[0] > self.dut_n_pixel[dut_index][0])  # Hits that are outside the x dimension of the DUT
            selection_y = np.logical_or(dut_hits.T[1] < 1, dut_hits.T[1] > self.dut_n_pixel[dut_index][1])  # Hits that are outside the y dimension of the DUT
            selection = np.logical_or(selection_x, selection_y)
            mask = np.zeros_like(dut_hits, dtype=np.bool)
            mask[selection] = 1

            # Mask hits due to inefficiency
            hit_indices = np.where(~selection)[0]  # Indices of not masked hits
            np.random.shuffle(hit_indices)  # shuffle these indeces
            n_inefficient_hit = int(hit_indices.shape[0] * (1. - self.dut_efficiencies[dut_index]))
            mask[hit_indices[:n_inefficient_hit]] = 1

            dut_hits = np.ma.array(dut_hits, dtype=np.uint16, mask=mask)
            digitized_hits.append(dut_hits)

        return digitized_hits

    def _create_data(self, start_event_number=0, n_events=10000):
        n_tracks_per_event = np.random.normal(self.tracks_per_event, self.tracks_per_event_sigma, n_events).astype(np.int)
        n_tracks_per_event[n_tracks_per_event < 0] = 0  # one cannot have less than 0 tracks per event
        track_positions_x, track_positions_y, track_angles_phi, track_angles_theta = self._create_tracks(n_tracks_per_event.sum())
        hits = self._create_hits_from_tracks(track_positions_x, track_positions_y, track_angles_phi, track_angles_theta)
        # Create event number
        events = np.arange(n_events)
        event_number = np.repeat(events, n_tracks_per_event).astype(np.int64)  # Create an event number of events with tracks
        event_number += start_event_number

        return event_number, self._digitize_hits(hits)

    def create_data_and_store(self, base_file_name, n_events, chunk_size=100000):
        logging.info('Simulate %d events with %d DUTs', n_events, self.n_duts)
        # Create output h5 files with emtpy hit ta
        output_files = []
        hit_tables = []
        for dut_index in range(self.n_duts):
            output_files.append(tb.open_file(base_file_name + '_DUT%d.h5' % dut_index, 'w'))
            hit_tables.append(output_files[dut_index].createTable(output_files[dut_index].root, name='Hits', description=self._hit_dtype, title='Simulated hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False)))

        progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=len(range(0, n_events, chunk_size)), term_width=80)
        progress_bar.start()
        # Fill output files in chunks
        for chunk_index, _ in enumerate(range(0, n_events, chunk_size)):
            actual_events, actual_digitized_hits = self._create_data(start_event_number=chunk_index * chunk_size, n_events=chunk_size)
            for dut_index in range(self.n_duts):
                actual_dut_hits = actual_digitized_hits[dut_index]
                selection = (actual_dut_hits.mask.T[0] == 0)  # Select not masked hits
                actual_hits = np.zeros(shape=np.count_nonzero(selection), dtype=self._hit_dtype)
                actual_hits['event_number'] = actual_events[selection]
                actual_hits['column'] = actual_dut_hits.T[0][selection]
                actual_hits['row'] = actual_dut_hits.T[1][selection]
                hit_tables[dut_index].append(actual_hits)
            progress_bar.update(chunk_index)
        progress_bar.finish()

        for output_file in output_files:
            output_file.close()


if __name__ == '__main__':
    simulate_data = SimulateData(1)
    simulate_data.create_data_and_store('simulated_data', n_events=1000000)


#     track_positions_x, track_positions_y, track_angles_phi, track_angles_theta = create_tracks(beam_position=0, beam_position_sigma=10000, beam_angle=0., beam_angle_sigma=2, n_tracks=10000)
#     intersections = intersect_tracks_with_duts(track_positions_x, track_positions_y, track_angles_phi, track_angles_theta, z_positions=[0, 10000])

    #     print track_angles_theta[0], track_angles_phi[1]
    #     print track_positions_x[0] + np.cos(track_angles_phi[0]) * np.sin(track_angles_theta[0]), track_positions_y[0] + np.sin(track_angles_phi[0]) * np.sin(track_angles_theta[0])
    #     plt.hist(track_angles_phi, bins=100)
    #     plt.show()
    #     plt.plot(track_positions_x, track_positions_y, '.')
    #     plt.plot(track_angles_theta, track_angles_phi, '.')
    #     plt.show()
