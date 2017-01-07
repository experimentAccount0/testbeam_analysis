''' Script to check that the examples run. The example data is reduced at the beginning to safe time.
'''
import os
import subprocess
import unittest
import tables as tb

import testbeam_analysis
from testbeam_analysis.tools import data_selection
from testbeam_analysis.examples import (eutelescope_example, fei4_telescope_example,
                                        simulated_data_example, telescope_with_time_reference_example)

package_path = os.path.dirname(testbeam_analysis.__file__)  # Get the absoulte path of the online_monitor installation
script_folder = os.path.abspath(os.path.join(package_path, r'examples/'))
tests_data_folder = os.path.abspath(os.path.join(os.path.realpath(script_folder), r'data/'))


def run_script_in_process(script, command=None):  # Run python script in blocking mode
    return subprocess.call(["%s" % 'python', script])


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.getenv('TRAVIS', False):
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()

        cls.output_folder = tests_data_folder

        # Reduce the example data to make it possible to test the examples in CI environments
        cls.examples_fei4_hit_files = [os.path.join(cls.output_folder, r'TestBeamData_FEI4_DUT0.h5'),
                                       os.path.join(cls.output_folder, r'TestBeamData_FEI4_DUT1.h5'),
                                       os.path.join(cls.output_folder, r'TestBeamData_FEI4_DUT4.h5'),
                                       os.path.join(cls.output_folder, r'TestBeamData_FEI4_DUT5.h5')]
        data_selection.reduce_hit_files(cls.examples_fei4_hit_files, fraction=100)

        cls.examples_mimosa_hit_files = [os.path.join(cls.output_folder, r'TestBeamData_Mimosa26_DUT%d.h5') % i for i in range(6)]
        data_selection.reduce_hit_files(cls.examples_mimosa_hit_files, fraction=100)

        # Remove old files and rename reduced files
        for file_name in cls.examples_fei4_hit_files:
            os.remove(file_name)
            os.rename(file_name[:-3] + '_reduced.h5', file_name)
        for file_name in cls.examples_mimosa_hit_files:
            os.remove(file_name)
            os.rename(file_name[:-3] + '_reduced.h5', file_name)
    @unittest.SkipTest # FIXME:
    def test_mimosa_example(self):
        eutelescope_example.run_analysis()

    @unittest.SkipTest # FIXME:
    def test_fei4_example(self):
        fei4_telescope_example.run_analysis()

    def test_simulated_data_example(self):
        ''' Check the example and the overall analysis that a efficiency of about 100% is reached.
            Not a perfect 100% is expected due to the finite propability that tracks are merged
            since > 2 tracks per event are simulated
        '''
        simulated_data_example.run_analysis(1000)
        with tb.open_file('simulation/Efficiency.h5') as in_file_h5:
            for dut_index in range(5):
                efficiency = in_file_h5.get_node('/DUT_%d/Efficiency' % dut_index)[:]
                efficiency_mask = in_file_h5.get_node('/DUT_%d/Efficiency_mask' % dut_index)[:]
                self.assertAlmostEqual(efficiency[~efficiency_mask].mean(), 100., delta=0.0001) 

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExamples)
    unittest.TextTestRunner(verbosity=2).run(suite)
