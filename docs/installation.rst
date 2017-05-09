Installation guide
==================

The following step-by-step installation guide explains in more details how you can setup *testbeam analysis*
under Windows, Linux or OS X operating system. A decent PC with at least 8 Gb of RAM is recommended!

Installing Miniconda Python
***************************

Miniconda Python is a Python distribution providing a simple to use package manager to install lots of Python packages for data analysis, plotting and I/O.
These packages are heavily used in *testbeam analysis*. Download and install Miniconda by following this link: "Continuum Miniconda":http://conda.pydata.org/miniconda.html. Choose Python 2.7.x according to your operating system architecture (e.g. 64-bit Python on 64-bit operating system)
&nbsp;
# *Installing C++ Compiler*
*Linux:* Install gcc via package manager, e.g. on Ubuntu run: <pre>sudo apt-get install build-essential</pre>*Windows:* Install "Microsoft Visual C++ Compiler for Python 2.7":http://aka.ms/vcpython27
*OS X:* Install "Xcode":https://itunes.apple.com/us/app/xcode/id497799835 from App Store and install Xcode Command Line Tools by running: <pre>xcode-select &#45;&#45;install</pre>
# *Installing Eclipse with PyDev*
Eclipse is a state of the art Integrated Development Environment (IDE) for almost every important software programming language (JAVA, C++, Python, Fortran, Ruby, Mathematica). It has a plug in system to extend the development environment. PyDev is the Python IDE plugin for Eclipse.
## Eclipse needs a Java runtime environmet (JRE) and will not start if this is not installed. *Windows:* It is recommend to use the Eclipse 32-bit version, even on a 64-bit machine. Since Eclipse 64-bit needs JRE 64-bit and Oracle JRE 64-bit does not provide automatic updates. This puts the PC at risk of viruses.
## Download the Eclipse from "Eclipse Homepage":http://www.eclipse.org/downloads. Eclipse does not need to be installed, the archive has to be extracted to a folder of your choice. Eclipse can be executed by double clicking on the executable.
## Eclipse asks for a workspace path where the projects will be located on your harddrive. Standard settings are sufficient.
## Close the welcome screen.
## Install the PyDev plugin by clicking on _Help -> Install New Software_ . Press add and fill the form (_name = PyDev_, _location = http://pydev.org/updates_):
&nbsp;
!{width:400px}PyDev.jpg!
&nbsp;
Select PyDev and install (accept license agreement and trust the certificate).
*Linux:* PyDev 3 is out and it needs Java 7, otherwise it will not show up in Eclipse without any error message. Please install java 7 and activate it. In Ubuntu activate it with <pre>sudo update-alternatives &#45;&#45;config java</pre>
## Add PyDev perspective to Eclipse and select it. The button is located in the upper rigth corner:
&nbsp;
!{width:300px}Perspective.jpg!
&nbsp;
## Goto _Window -> Preferences -> PyDev -> Interpreters-> Python Interpreter_ and press new.
&nbsp;
!{width:600px}AnacondaSetup1.jpg!
&nbsp;
Select the Python executable in /home/<username>/anaconda/bin/ on *Linux* or c:\Anaconda\ on *Windows* (optionally use the Anaconda/env/<environment name>/ folder if you are using Anaconda environments) and press the OK button. Everything is set up automatically.
More details are given "here":http://docs.continuum.io/anaconda/ide_integration.html.
&nbsp;
# *Installing Required Python Packages*
Open a console and type one by one:
<pre>conda update conda</pre><pre>conda install numpy cython pytables scipy matplotlib pandas pyserial bitarray nose pyzmq pyyaml</pre><pre>pip install progressbar-latest pyvisa pyvisa-py pyqtgraph mock</pre>On *Windows* additionally run:
<pre>conda install pywin32</pre>
# *Installing pyTestbeamAnalysis*
## Goto File->Import->Git and select Projects from Git
## Click clone URI and type the pyTestbeamAnalysis repository (_URI: https://github.com/SiLab-Bonn/pyTestbeamAnalysis_)
## If you have a "GitHub":https://github.com account you can add the credentials here
## Click next, select all branches, click next and specify the directory where pyBAR will be cloned to
## Wait until the download of the code is finished and check _Check out as project configured using the New Project Wizard_
## Select _PyDev -> PyDevProject_
## Give the project a name, select the folder where the pyBAR was cloned to (e.g. /home/username/git/pyBAR) and click finish
## Open a shell and run the following command from the pyBar/host folder: <pre> python setup.py develop </pre> This will compile and install pyTestbeamAnalysis to the environment.
*Windows:* If the compilation fails use the Visual Studio Command promt to run the setup script. Because distutils sometimes cannot find VS due to multiple/old VS installation. It might also be needed to install and "activate the 64-bit compiler":https://msdn.microsoft.com/en-us/library/x4d2c09s%28v=vs.90%29.aspx if you use 64-bit.