# Testbeam_Analysis [![BuildStatus](https://travis-ci.org/SiLab-Bonn/testbeam_analysis.svg?branch=development)](https://travis-ci.org/SiLab-Bonn/testbeam_analysis) [![Build Status](https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/testbeam_analysis)](https://ci.appveyor.com/project/DavidLP/testbeam-analysis) [![Coverage Status](https://coveralls.io/repos/SiLab-Bonn/testbeam_analysis/badge.svg?branch=development&service=github)](https://coveralls.io/github/SiLab-Bonn/testbeam_analysis?branch=development)
A simple analysis of pixel-sensors data in particle beams. All steps of a complete analysis are implemented with a few independent python functions. If you want to do simple straight line fits without a Kalman filter or you want to understand the basics of telescope dara reconstruction this code might help. 
If you want to have something fancy to account for thick devices in combination with low energetic beams use e.g. _EUTelescope_. Depending on the setup a resolution that is only ~ 15% worse can be archieved with this code.
For a quick first impression check the example plots in the wiki.

In future releases it is forseen to make the code more readable and to implement a Kalman Filter to have the best possible track fit results.

# Installation
You have to have Python 2/3 with the following modules installed:
- cython
- tables
- scipy
- matplotlib
- numba

If you are new to Python please look at the installation guide in the wiki.
Since it is recommended to change example files according to your needs you should install the module with
```bash
python setup.py develop
```
This does not copy the code to a new location, but just links to it.
Uninstall:
```bash
pip uninstall testbeam_analysis
```

# Example usage
Check the examples folder with data and examples of a Mimosa26 and a FE-I4 telescope analysis.
Run eutelescope_example.py or fei4_telescope_example.py in the example folder and check the text output to the console as well as the plot and data files that are created to understand what is going on.
In the examples folder type e.g.:
```bash
python fei4_telescope_example.py
```



