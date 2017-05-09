===============================================
Introduction
===============================================

|travis-status|  |appveyor-status|  |doc|  |coverage|

Testbeam analysis is a simple analysis of pixel-sensor data in particle beams. All steps of a complete analysis
are implemented with a few independent python functions. If you want to understand the basics of telescope data
reconstruction this code might help. 
If you want to have something fancy to account for thick devices in combination with low energetic beams
use e.g. _EUTelescope_. Depending on the setup a resolution that is only ~ 15% worse can be archieved with this code.
For a quick first impression check the examples in the documentation.

In future releases it is forseen to make the code more readable and to implement a Kalman Filter to have the best
possible track fit results.

Installation
============
You have to have Python 2/3 with the following modules installed:
- cython
- tables
- scipy
- matplotlib
- numba

If you are new to Python please look at the installation guide in the wiki.
Since it is recommended to change example files according to your needs you should install the module with

.. code-block:: bash

   python setup.py develop

This does not copy the code to a new location, but just links to it.
Uninstall:

.. code-block:: bash

   pip uninstall testbeam_analysis


Example usage
==============
Check the examples folder with data and examples of a Mimosa26 and a FE-I4 telescope analysis.
Run eutelescope_example.py or fei4_telescope_example.py in the example folder and check the text output to
the console as well as the plot and data files that are created to understand what is going on.
In the examples folder type e.g.:

.. code-block:: bash
   
   python fei4_telescope_example.py

.. |travis-status| image:: https://travis-ci.org/SiLab-Bonn/testbeam_analysis.svg?branch=development
    :target: https://travis-ci.org/SiLab-Bonn/testbeam_analysis
    :alt: Build status
    
.. |appveyor-status| image:: https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/testbeam_analysis
    :target: https://ci.appveyor.com/project/DavidLP/testbeam-analysis
    :alt: Build status

.. |doc| image:: https://img.shields.io/badge/documentation--green.svg
    :target: http://silab-bonn.github.io/testbeam_analysis
    :alt: Documentation
    
.. |coverage| image:: https://coveralls.io/repos/SiLab-Bonn/testbeam_analysis/badge.svg?branch=development
    :target: https://coveralls.io/github/SiLab-Bonn/testbeam_analysis?branch=development
    :alt: Coverage


