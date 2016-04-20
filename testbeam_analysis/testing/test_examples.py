''' Script to check that the examples run. The example data is reduced at the beginning to safe time.
'''
import os
import subprocess
import unittest

import testbeam_analysis
from testbeam_analysis.tools import data_selection

package_path = os.path.dirname(testbeam_analysis.__file__)  # Get the absoulte path of the online_monitor installation
script_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(package_path)) + r'/examples'))
tests_data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(package_path)) + r'/examples/data'))


def run_script_in_process(script, command=None):  # Run python script in blocking mode
    return subprocess.call(["%s" % 'python', script])


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if os.name != 'nt':
            try:
                from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
                cls.vdisplay = Xvfb()
                cls.vdisplay.start()
            except (ImportError, EnvironmentError):
                pass

        cls.output_folder = tests_data_folder

        # Reduce the example data to make it possible to test the examples in CI environments
        cls.examples_fei4_hit_files = [cls.output_folder + r'\TestBeamData_FEI4_DUT0.h5',
                                       cls.output_folder + r'\TestBeamData_FEI4_DUT1.h5',
                                       cls.output_folder + r'\TestBeamData_FEI4_DUT4.h5',
                                       cls.output_folder + r'\TestBeamData_FEI4_DUT5.h5']
        data_selection.reduce_hit_files(cls.examples_fei4_hit_files, fraction=100)

        cls.examples_mimosa_hit_files = [cls.output_folder + r'\TestBeamData_Mimosa26_DUT%d.h5' % i for i in range(6)]
        data_selection.reduce_hit_files(cls.examples_mimosa_hit_files, fraction=100)

        # Remove old files and rename reduced files
        for file_name in cls.examples_fei4_hit_files:
            os.remove(file_name)
            os.rename(file_name[:-3] + '_reduced.h5', file_name)
        for file_name in cls.examples_mimosa_hit_files:
            os.remove(file_name)
            os.rename(file_name[:-3] + '_reduced.h5', file_name)

    def test_mimosa_example(self):
        return_value = run_script_in_process(script_folder + r'\eutelescope_example.py')
        self.assertEqual(return_value, 0)

    def test_fei4_example(self):
        return_value = run_script_in_process(script_folder + r'\fei4_telescope_example.py')
        self.assertEqual(return_value, 0)

    def test_simulated_data_example(self):
        return_value = run_script_in_process(script_folder + r'\simulated_data_example.py')
        self.assertEqual(return_value, 0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExamples)
    unittest.TextTestRunner(verbosity=2).run(suite)
