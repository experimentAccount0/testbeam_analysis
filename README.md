# pyTestbeamAnalysis [[![BuildStatus](https://travis-ci.org/SiLab-Bonn/testbeam_analysis.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/testbeam_analysis) [![Build Status](https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/testbeam_analysis)](https://ci.appveyor.com/project/DavidLP/testbeam_analysis)
A _very_ simple analysis of pixel-sensor data from test beams. All steps of a full analysis are included in very few lines of Python code. If you want to do simple straight line fits without a Kalman filter or you want to understand the basics of telescope reconstruction this code might help. 
If you want to have something fancy to account for thick devices in combination with low energetic beams use e.g. _EUTelescope_. Depending on the setup a resolution that is only ~ 15% worse can be archieved with this code.
For a quick first impression check the example plots in the wiki.

In future releases it is forseen to make the code more readable and to implement a Kalman Filter to have the best possible track fit results.

# Installation
You have to have Python 2.7 and the following modules installed:
- cython
- pandas
- tables
- scipy
- matplotlib

If you are new to Python please look at the installation guide in the wiki.
Since it is recommended to change the one and only file according to your needs you should install the module with
```bash
python setup.py develop
```
This does not copy the code to a new location, but just links to it.
Uninstall:
```bash
pip uninstall pyTestbeamAnalysis
```

# Example usage
Check the examples folder with data and examples of a Mimosa26 and a FE-I4 telescope analysis.
Run eutelescope_example.py or fei4_telescope_example.py in the example folder and check the text output to the console as well as the plot and data files that are created to understand what is going on.
In the examples folder type e.g.:
```bash
python fei4_telescope_example.py
```



