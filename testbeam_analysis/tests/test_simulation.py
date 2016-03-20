''' Script to test the simulator.
'''
import os
import tables as tb
import numpy as np
import unittest
from pyLandau import landau
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from testbeam_analysis.tools import simulate_data
from testbeam_analysis import geometry_utils

from testbeam_analysis import analysis_utils
from testbeam_analysis import plot_utils


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

        self.simulate_data = simulate_data.SimulateData(0)
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
        self.simulate_data.dut_material_budget = [0] * self.simulate_data.n_duts  # Turn off multiple scattering
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

    def test_multiple_scattering(self):
        self.simulate_data.reset()

        # Set two planes and check the scattering angle due to the material budget of the first plane
        self.simulate_data.n_duts = 2
        self.simulate_data.set_std_settings()
        self.simulate_data.beam_angle = 0
        self.simulate_data.beam_angle_sigma = 0
        self.simulate_data.beam_position_sigma = (0, 0)
        self.simulate_data.z_positions = [i * 1000000 + 1000 for i in range(self.simulate_data.n_duts)]  # 1m distance to see scattering better
        self.simulate_data.dut_pixel_size = [(1, 1)] * self.simulate_data.n_duts  # If the pixel size is too big this tests fails due to pixel discretisation error
        self.simulate_data.dut_n_pixel = [(10000, 10000)] * self.simulate_data.n_duts

        self.simulate_data.digitization_charge_sharing = False  # Simplify position reconstruction

        def gauss(x, A, mu, sigma):  # Scattering angle theta fit function
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        def check_scattering_angle():
            # Expected scattering angle theta_0
            theta_0 = self.simulate_data._scattering_angle_sigma(self.simulate_data.dut_material_budget[0], charge_number=1) * 1000

            # Extract theta from simulation results by using the hit positions
            with tb.open_file('simulated_data_DUT0.h5', 'r') as in_file_h5:
                dut0_x = (in_file_h5.root.Hits[:]['column'] - 1) * self.simulate_data.dut_pixel_size[0][0]
                dut0_y = (in_file_h5.root.Hits[:]['row'] - 1) * self.simulate_data.dut_pixel_size[0][1]

            with tb.open_file('simulated_data_DUT1.h5', 'r') as in_file_h5:
                dut1_x = (in_file_h5.root.Hits[:]['column'] - 1) * self.simulate_data.dut_pixel_size[1][0]
                dut1_y = (in_file_h5.root.Hits[:]['row'] - 1) * self.simulate_data.dut_pixel_size[1][1]

            # Calculate theta in spherical coordinates from data
            dx = dut1_x.astype(np.float) - dut0_x.astype(np.int)
            dy = dut1_y.astype(np.float) - dut0_y.astype(np.float)
            dz = np.ones_like(dx) * (self.simulate_data.z_positions[1] - self.simulate_data.z_positions[0])
            _, theta, _ = geometry_utils.cartesian_to_spherical(dx, dy, dz)

            if theta_0 == 0:  # Special case: no material budget thus not scattering, theta has to be beam angle
                self.assertTrue(np.allclose(self.simulate_data.beam_angle, theta * 1000., atol=0.01))
                return

            if self.simulate_data.beam_angle == 0:
                hist, bins = np.histogram(theta * 1000, range=(0, np.pi), bins=1000)  # Histogramm scattering angle distribution
                x, y = (bins[:-1] + bins[1:]) / 2., hist
#                 import matplotlib.pyplot as plt
#                 plt.bar(x, y, width=np.diff(x)[0])
#                 plt.plot(x, gauss(x, *(np.amax(y), 0, theta_0)), 'r-', linewidth=2)
#                 plt.legend()
#                 plt.show()
                # Fit scatterign distribution
                coeff, _ = curve_fit(gauss, x, y, p0=[np.amax(hist), 0, theta_0])  # Fit theta distribution
                self.assertTrue(np.allclose(theta_0, coeff[2], atol=0.02), 'The scattering angle for multiple scattering is wrong')  # Check for similarity, slight differences most likely due to pixel position binning
            else:  # TODO: this is maybe a bug, the theta distribution with starting beam angle does not look gaussian; just check the mean here;
                self.assertTrue(np.allclose(np.mean(theta * 1000.), self.simulate_data.beam_angle, atol=0.05), 'The beam direction with multiple scattering is wrong')

        # Test 1: Check scattering for different device thickness from 50 um to 1000 um
        for device_thickness in range(0, 1000, 50):  # Change the thickness
            self.simulate_data.dut_thickness = [device_thickness] * self.simulate_data.n_duts
            self.simulate_data.dut_material_budget = [self.simulate_data.dut_thickness[i] * 1e-4 / 9.370 for i in range(self.simulate_data.n_duts)]  # Assume silicon sensor
            self.simulate_data.create_data_and_store('simulated_data', n_events=100000)
            check_scattering_angle()

        # Test 2: Check scattering for different device z positions
        for z_position in ([[i, i + 1000000] for i in range(0, 1000000, 250000)]):  # Put planes at different positions
            self.simulate_data.z_positions = z_position
            self.simulate_data.create_data_and_store('simulated_data', n_events=100000)
            check_scattering_angle()

        # Test 3: Check with beam angle
        for beam_angle in range(5):
            self.simulate_data.beam_angle = beam_angle
            self.simulate_data.z_positions = [0, 100000]
            self.simulate_data.dut_thickness = [1000] * self.simulate_data.n_duts
            self.simulate_data.dut_material_budget = [self.simulate_data.dut_thickness[i] * 1e-4 / 9.370 for i in range(self.simulate_data.n_duts)]
            self.simulate_data.create_data_and_store('simulated_data', n_events=100000)
            check_scattering_angle()
    #@unittest.SkipTest

    def test_dut_rotation(self):
        self.simulate_data.reset()

        # Set two planes and check the scattering angle due to the material budget of the first plane
        self.simulate_data.n_duts = 2
        self.simulate_data.set_std_settings()
        self.simulate_data.tracks_per_event_sigma = 0
        self.simulate_data.beam_angle = 0
        self.simulate_data.beam_angle_sigma = 0
        self.simulate_data.dut_pixel_size = [(1, 1)] * self.simulate_data.n_duts  # If the pixel size is too big this tests fails due to pixel discretisation error
        self.simulate_data.dut_n_pixel = [(10000, 10000)] * self.simulate_data.n_duts
        self.simulate_data.tracks_per_event = 1
        self.simulate_data.tracks_per_event_sigma = 0
        self.simulate_data.digitization_charge_sharing = False  # Not needed, save time
        self.dut_material_budget = [0] * self.simulate_data.n_duts  # Turn off multiple scattering

        def check_rotations():  # Rotations are checked by column / row correlation of the devices and analysing these correlation
            def line(x, c0, c1):  # Correlation line fit
                return c1 * x + c0
            with tb.open_file('simulated_data_DUT0.h5', 'r') as in_file_1_h5:
                with tb.open_file('simulated_data_DUT1.h5', 'r') as in_file_2_h5:
                    # Merge data on event basis for correlation, since one cannot assume that every plane is always hit
                    hits_0, hits_1 = analysis_utils.merge_on_event_number(in_file_1_h5.root.Hits[:], in_file_2_h5.root.Hits[:])

                    column_0, column_1 = hits_0['column'], hits_1['column']
                    row_0, row_1 = hits_0['row'], hits_1['row']

                    coeff_column, error_column = curve_fit(line, column_1, column_0, p0=[0, 1.])
                    coeff_row, error_row = curve_fit(line, row_1, row_0, p0=[0, 1.])

#                     plt.plot(row_1, row_0, '.')
#                     plt.plot(column_1, column_0, '.')
#                     plt.plot(column_1, line(column_1, *coeff_column), '-', label='column')
#                     plt.plot(row_1, line(row_1, *coeff_row), '-', label='row')
#                     plt.xlabel('DUT 1')
#                     plt.ylabel('DUT 0')
#                     plt.legend(loc=0)
#                     plt.show()

                    if coeff_column[1] > 1 and np.allclose(coeff_column[1], 1., atol=0.01):  # Can happen if first DUT is rotated or fitting precision
                        coeff_column[1] = 1

                    if coeff_row[1] > 1 and np.allclose(coeff_row[1], 1., atol=0.01):  # Can happen if first DUT is rotated or fitting precision
                        coeff_row[1] = 1

#                     print 'offset column', coeff_column[0]
#                     print 'want column', self.simulate_data.offsets[1][0] - self.simulate_data.offsets[0][0]
#                     print 'slope column', coeff_column[1]
#                     print 'want column', self.simulate_data.rotations[1][1]
#                     print 'have column', np.arccos(coeff_column[1])
#                     print 'slope row', coeff_row[1]
#                     print 'want row', self.simulate_data.rotations[1][0]
#                     print 'have row', np.arccos(coeff_row[1])

                    # Check column / row relative offsets from line fit offsets with 1 mu precision
                    self.assertAlmostEqual(self.simulate_data.offsets[1][0] - self.simulate_data.offsets[0][0], coeff_column[0], delta=1.)  # Check with 10 mRad precision
                    self.assertAlmostEqual(self.simulate_data.offsets[1][1] - self.simulate_data.offsets[0][1], coeff_row[0], delta=1.)  # Check with 10 mRad precision
                    # Check alpha / beta angles with from line fit slopes with 10 mRad precision
                    self.assertAlmostEqual(self.simulate_data.rotations[1][1], np.arccos(coeff_column[1]), delta=0.01)  # Check with 10 mRad precision
                    self.assertAlmostEqual(self.simulate_data.rotations[1][0], np.arccos(coeff_row[1]), delta=0.01)  # Check with 10 mRad precision

        # Test 1: Check correlation for different alpha angles (x-axis rotation) and different relative offsets in x/y between the planes
        for alpha in [0, np.pi / 8., np.pi / 6., np.pi / 4., np.pi / 3.]:
            for offset_x in range(-1000, 100, 1001):
                for offset_y in range(-1000, 100, 1001):
                    self.simulate_data.rotations[1] = (alpha, 0., 0.)
                    self.simulate_data.offsets[1] = (self.simulate_data.offsets[0][0] + offset_x, self.simulate_data.offsets[0][1] + offset_y)  # Set x/y shift with respect to DUT 0
                    self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
                    check_rotations()

        # Test 2: Check correlation for different beta angles (y-axis rotation) and different relative offsets in x/y between the planes
        for beta in [0, np.pi / 8., np.pi / 6., np.pi / 4., np.pi / 3.]:
            for offset_x in range(-1000, 100, 1001):
                for offset_y in range(-1000, 100, 1001):
                    self.simulate_data.rotations[1] = (0., beta, 0.)
                    self.simulate_data.offsets[1] = (self.simulate_data.offsets[0][0] + offset_x, self.simulate_data.offsets[0][1] + offset_y)  # Set x/y shift with respect to DUT 0
                    self.simulate_data.create_data_and_store('simulated_data', n_events=10000)
                    check_rotations()


#         with tb.open_file('simulated_data_DUT0.h5', 'r') as in_file_1_h5:
#             with tb.open_file('simulated_data_DUT1.h5', 'r') as in_file_2_h5:
#                 hits_1, hits_2 = analysis_utils.merge_on_event_number(in_file_1_h5.root.Hits[:], in_file_2_h5.root.Hits[:])
#
#                 column_1, column_2 = hits_1['column'], hits_2['column']
#                 row_1, row_2 = hits_1['row'], hits_2['row']
#
#                 plt.plot(row_2, row_1, '.')
#                 plt.plot(column_2, column_1, '.')
# #
#                 coeff, err = curve_fit(line, column_2, column_1, p0=[0, 1.])
#                 plt.plot(column_2, line(column_2, *coeff), '-', label='column')
#
#                 print 'slope', coeff[1]
#                 print 'want', self.simulate_data.rotations[1][1]
#                 print np.arccos(coeff[1])
#
#                 coeff, err = curve_fit(line, row_2, row_1, p0=[0, 1.])
#                 plt.plot(row_2, line(row_2, *coeff), '-', label='row')
# #
#                 plt.xlabel('DUT 1')
#                 plt.ylabel('DUT 0')
#                 plt.legend(loc=0)
#
#                 print 'slope', coeff[1]
#                 print 'want', self.simulate_data.rotations[1][0]
#                 print 'have', np.arccos(coeff[1])

#                 print column_1.shape
#                 fig = plt.gca()
#                 hist2d = analysis_utils.hist_2d_index(column_2, row_2, shape=self.simulate_data.dut_n_pixel[0])
#                 extent = [0.5, self.simulate_data.dut_n_pixel[0][0] + .5, self.simulate_data.dut_n_pixel[0][1] + .5, 0.5]
#                 im = plt.imshow(hist2d, interpolation='none', aspect="auto")

#                 print 'offset', coeff[0]
#                 print 'have', self.simulate_data.offsets[1][0] / ( 1.- np.arccos(coeff[1]))

#                 plt.show()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
