''' Script to test the simulator.
'''
import os
import tables as tb
import numpy as np
import unittest
from pyLandau import landau

from testbeam_analysis.tools.simulate_data import SimulateData


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if os.name != 'nt':
            try:
                from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
                self.vdisplay = Xvfb()
                self.vdisplay.start()
            except (ImportError, EnvironmentError):
                pass

        self.simulate_data = SimulateData(0)
        self.simulate_data.n_duts = 6

    @classmethod
    def tearDownClass(self):  # remove created files
        for dut_index in range(self.simulate_data.n_duts):
            os.remove('simulated_data_DUT%d.h5' % dut_index)

    def test_position(self):  # Test beam position with respect to devices positions
        self.simulate_data.reset()

        def check_position():  # Helper function to be called with different position parameter data
            # Calculate expectation
            expected_mean_column, expected_mean_row = [], []
            for dut_index in range(self.simulate_data.n_duts):
                expected_mean_column.append(self.simulate_data.beam_position[0] - self.simulate_data.offsets[dut_index][0])
                expected_mean_row.append(self.simulate_data.beam_position[1] - self.simulate_data.offsets[dut_index][1])
            # Extract results
            mean_column, mean_row = [], []
            for dut_index in range(self.simulate_data.n_duts):
                with tb.open_file('simulated_data_DUT%d.h5' % dut_index, 'r') as in_file_h5:
                    mean_column.append((in_file_h5.root.Hits[:]['column'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][0])
                    mean_row.append((in_file_h5.root.Hits[:]['row'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][1])

            self.assertTrue(np.allclose(expected_mean_column, mean_column, rtol=0.01, atol=10))
            self.assertTrue(np.allclose(expected_mean_row, mean_row, rtol=0.01, atol=10))

        # Test 1: Check different DUT offsets
        self.simulate_data.offsets = [(-35000, -35000), (-30000, -30000), (-25000, -25000), (-20000, -20000), (-15000, -15000), (-10000, -10000)]  # Set DUT offsets with respect to beam
        self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
        check_position()

        # Test 2: Check different beam offset
        self.simulate_data.beam_position = (500, 500)  # Shift beam position
        self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
        check_position()

        # Test 3: Check different beam parameter
        self.simulate_data.beam_position_sigma = (0, 0)  # beam position sigma
        self.simulate_data.beam_position = (0, 0)  # Shift beam position
        self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
        check_position()

    def test_charge_distribution(self):
        self.simulate_data.reset()

        def check_charge():  # Helper function to be called with different charge parameter data
            for dut_index in range(self.simulate_data.n_duts):  # Loop over DUTs
                mpv_charge = 77 * self.simulate_data.dut_thickness[dut_index]  # 77 electrons per um in silicon
                x = np.arange(0, 10, 0.1) * mpv_charge
                if self.simulate_data.dut_noise[dut_index]:
                    y = landau.langau(x, mu=mpv_charge, eta=0.2 * mpv_charge, sigma=self.simulate_data.dut_noise[dut_index])
                else:
                    y = landau.landau(x, mu=mpv_charge, eta=0.2 * mpv_charge)
                with tb.open_file('simulated_data_DUT%d.h5' % dut_index, 'r') as in_file_h5:
                    charge = in_file_h5.root.Hits[:]['charge'] * 10.  # 1 LSB corresponds to 10 electrons
                    charge_hist, edges = np.histogram(charge, bins=100, range=(x[0], x[-1]))
#                     import matplotlib.pyplot as plt
#                     plt.plot(edges[:-1], charge_hist)
#                     plt.plot(x, y / np.amax(y) * np.amax(charge_hist))
#                     plt.xlim((0, 100000))
#                     plt.show()
                    self.assertTrue(np.allclose(y / np.amax(y) * np.amax(charge_hist), charge_hist, rtol=1, atol=10))

        # Check Landau for different device thickness
        self.simulate_data.dut_thickness = [(i + 1) * 100 for i in range(self.simulate_data.n_duts)]  # Create charge distribution in different device thicknesses, thus Landau MPW should change
        self.simulate_data.digitization_charge_sharing = False  # To judge deposited charge, charge sharing has to be off
        self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
        check_charge()

        # Check Landau for different device noiose
        self.simulate_data.reset()
        self.simulate_data.dut_thickness = [200 for i in range(self.simulate_data.n_duts)]  # Fix device thickness
        self.simulate_data.dut_noise = [i * 1000 for i in range(self.simulate_data.n_duts)]  # Create charge distribution in different device noise, thus Langau sigma should change
        self.simulate_data.digitization_charge_sharing = False  # To judge deposited charge, charge sharing has to be off
        self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
        check_charge()

    def test_beam_angle(self):
        self.simulate_data.reset()

        def check_beam_angle():
            # Expected offsets in x, y at DUT planes due to initial beam angle theta and direction distribution phi = (start, stop)
            expected_offsets_x, expected_offsets_y = [], []
            if self.simulate_data.beam_direction[0] < self.simulate_data.beam_direction[1]:
                mean_direction_cos = np.cos(np.arange(self.simulate_data.beam_direction[0], self.simulate_data.beam_direction[1], 0.01)).mean()  # A mean angle does not translate linearly to a mean offset
                mean_direction_sin = np.sin(np.arange(self.simulate_data.beam_direction[0], self.simulate_data.beam_direction[1], 0.01)).mean()
            else:
                mean_direction_cos = np.cos(self.simulate_data.beam_direction[0])
                mean_direction_sin = np.sin(self.simulate_data.beam_direction[0])
            for dut_index in range(self.simulate_data.n_duts):
                offset = self.simulate_data.beam_position[0] - self.simulate_data.offsets[dut_index][0]
                expected_offsets_x.append(offset + mean_direction_cos * np.tan(self.simulate_data.beam_angle / 1000.) * self.simulate_data.z_positions[dut_index])
                expected_offsets_y.append(offset + mean_direction_sin * np.tan(self.simulate_data.beam_angle / 1000.) * self.simulate_data.z_positions[dut_index])

            # Extract results
            mean_column, mean_row = [], []
            for dut_index in range(self.simulate_data.n_duts):
                with tb.open_file('simulated_data_DUT%d.h5' % dut_index, 'r') as in_file_h5:
                    mean_column.append((in_file_h5.root.Hits[:]['column'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][0])
                    mean_row.append((in_file_h5.root.Hits[:]['row'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][1])

            # Check for similarity, on pixel width error expected (binning error)
            self.assertTrue(np.allclose(expected_offsets_x, mean_column, rtol=0.001, atol=self.simulate_data.dut_pixel_size[0][0]))
            self.assertTrue(np.allclose(expected_offsets_y, mean_row, rtol=0.001, atol=self.simulate_data.dut_pixel_size[0][0]))

        # Test 1: Fixed theta angle, different fixed phi
        self.simulate_data.beam_angle = 5  # If the angle is too small this tests fails due to pixel discretisation error
        self.simulate_data.dut_pixel_size = [(1, 1)] * self.simulate_data.n_duts  # If the pixel size is too big this tests fails due to pixel discretisation error
        self.simulate_data.dut_n_pixel = [(10000, 10000)] * self.simulate_data.n_duts  # If the sensor is too small the mean cannot be easily calculated
        self.simulate_data.beam_angle_sigma = 0
        self.simulate_data.beam_position_sigma = (0, 0)
        self.simulate_data.dut_material_budget = [0] * self.simulate_data.n_duts  # Turn off mulitple scattering
        self.simulate_data.digitization_charge_sharing = False  # Simplify position reconstruction

        for phi in [0, np.pi / 4., np.pi / 2., 3. / 4. * np.pi, np.pi, 5. * np.pi / 4., 3 * np.pi / 2.]:
            self.simulate_data.beam_direction = (phi, phi)
            self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
            check_beam_angle()

        # Test 2: Fixed theta angle, different phi ranges
        for phi_max in [0, np.pi / 4., np.pi / 2., 3. / 4. * np.pi, np.pi, 5. * np.pi / 4., 3 * np.pi / 2.]:
            self.simulate_data.beam_direction = (0, phi_max)
            self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
            check_beam_angle()

        # Test 3: Fixed theta angle, different phi ranges
        for phi_max in [0, np.pi / 4., np.pi / 2., 3. / 4. * np.pi, np.pi, 5. * np.pi / 4., 3 * np.pi / 2.]:
            self.simulate_data.beam_direction = (0, phi_max)
            self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
            check_beam_angle()

        # Test 4: Gaussian dstributed theta angle, full phi range
        self.simulate_data.beam_angle_sigma = 2
        self.simulate_data.beam_direction = (0, 2. * np.pi)
        self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
        check_beam_angle()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
