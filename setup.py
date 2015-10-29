#!/usr/bin/env python
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
import numpy as np
from Cython.Build import cythonize

cpp_extension = cythonize([
    Extension('testbeam_analysis.hit_clusterizer', ['testbeam_analysis/clusterizer/hit_clusterizer.pyx', 'testbeam_analysis/clusterizer/Clusterizer.cpp', 'testbeam_analysis/clusterizer/Basis.cpp']),
    Extension('testbeam_analysis.analysis_functions', ['testbeam_analysis/clusterizer/analysis_functions.pyx'])
])

author = 'Christian Bespin, David-Leon Pohl'
author_email = 'christian.bespin@uni-bonn.de, pohl@physik.uni-bonn.de'

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

f = open('VERSION', 'r')
version = f.readline().strip()
f.close()

setup(
    name='testbeam_analysis',
    version=version,
    description='A light weight test beam analysis in Python and C++.',
    url='https://github.com/SiLab-Bonn/testbeam_analysis',
    license='BSD 3-Clause ("BSD New" or "BSD Simplified") License',
    long_description='A very simple analysis of pixel-sensor data from testbeams. All steps of a full analysis are included in one file in < 1500 lines of Python code. If you you want to do simple straight line fits without a Kalman filter or you want to understand the basics of telescope reconstruction this code might help. If you want to have something fancy to account for thick devices in combination with low energetic beams use e.g. EUTelescope. Depending on the setup a resolution that is only ~ 15% worse can be archieved with this code. For a quick first impression check the example plots in the wiki and run the examples.',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),  # exclude=['*.tests', '*.test']),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['*.txt', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cpp_extension,
    include_dirs=[np.get_include()],
    keywords=['testbeam', 'particle', 'reconstruction', 'pixel', 'detector'],
    platforms='any'
)
