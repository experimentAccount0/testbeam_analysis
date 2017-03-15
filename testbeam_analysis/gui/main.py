import sys
import logging
from email import message_from_string
from pkg_resources import get_distribution, DistributionNotFound

from PyQt5 import QtCore, QtWidgets, QtGui

from data import DataTab
from setup import SetupTab
from settings import SettingsWindow

import testbeam_analysis
from testbeam_analysis.gui import tab_widget

PROJECT_NAME = 'Testbeam Analysis'
GUI_AUTHORS = 'Pascal Wolf, David-Leon Pohl'
MINIMUM_RESOLUTION = (1366, 768)

try:
    pkgInfo = get_distribution(
        'testbeam_analysis').get_metadata('PKG-INFO')
    for value in message_from_string(pkgInfo).items():
        if value[0] == 'Author':
            AUTHORS = value[1]
except DistributionNotFound:
    AUTHORS = 'Not defined'


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """ TODO: DOCSTRING"""
        super(AnalysisWindow, self).__init__(parent)

        # Get default settings
        self.setup = SettingsWindow().default_setup
        self.options = SettingsWindow().default_options

        # Make variable for SettingsWindow
        self.sw = None

        # Make dict to access tab widgets
        self.tw = {}

        self._init_UI()

    def _init_UI(self):
        """ TODO: DOCSTRING"""

        # Main window settings
        self.setWindowTitle(PROJECT_NAME)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setMinimumSize(MINIMUM_RESOLUTION[0], MINIMUM_RESOLUTION[1])
        self.resize(0.75 * self.screen.width(), 0.75 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Add widgets to main window
        self._init_menu()
        self._init_tabs()

        self.handle_messages("Hello and welcome to a simple and easy to use testbeam analysis!", 4000)

    def _init_tabs(self):
        # Add tab_widget and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'Noisy Pixel', 'Clustering',
                          'Correlations', 'Pre-alignment', 'Track finding',
                          'Alignment', 'Track fitting', 'Analysis', 'Result')

        # Add QTabWidget for tab_widget
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize each tab
        for i, name in enumerate(self.tab_order):
            if name == 'Files':
                widget = DataTab(parent=self.tabs)
            elif name == 'Setup':
                widget = SetupTab(parent=self.tabs)
            elif name == 'Noisy Pixel':
                widget = tab_widget.NoisyPixelsTab(parent=self.tabs,
                                                   setup=self.setup,
                                                   options=self.options)
            elif name == 'Clustering':
                widget = tab_widget.ClusterPixelsTab(parent=self.tabs,
                                                     setup=self.setup,
                                                     options=self.options)
            elif name == 'Pre-alignment':
                widget = tab_widget.PrealignmentTab(parent=self.tabs,
                                                    setup=self.setup,
                                                    options=self.options)
            elif name == 'Track finding':
                widget = tab_widget.TrackFindingTab(parent=self.tabs,
                                                    setup=self.setup,
                                                    options=self.options)
            elif name == 'Alignment':
                widget = tab_widget.AlignmentTab(parent=self.tabs,
                                                 setup=self.setup,
                                                 options=self.options)
            elif name == 'Track fitting':
                widget = tab_widget.TrackFittingTab(parent=self.tabs,
                                                    setup=self.setup,
                                                    options=self.options)
            elif name == 'Result':
                widget = tab_widget.ResultTab(parent=self.tabs,
                                              setup=self.setup,
                                              options=self.options)
            else:
                logging.info('GUI for %s not implemented yet', name)
                continue

            self.tw[name] = widget
            self.tabs.addTab(self.tw[name], name)

            # Disable all tabs but DataTab. Enable tabs later via self.enable_tabs()
            if i > 0:
                self.tabs.setTabEnabled(i, False)

            # Disable all widgets of all tabs but DataTab
            # if name != 'Files':
            # self.tw[name].setDisabled(True)

        # Connect signals in between tabs and main window

        # Connect DataTab
        self.tw['Files'].proceedAnalysis.connect(lambda: self.tw['Setup'].input_data(self.tw['Files'].data))

        # Connect SetupTab
        self.tw['Setup'].proceedAnalysis.connect(lambda: self.update_data(self.tw['Setup'].data))

        # Connect statusMessage and proceedAnalysis signal of all tabs
        for name in self.tab_order:
            try:
                self.tw[name].statusMessage.connect(lambda message: self.handle_messages(message, 4000))
                self.tw[name].proceedAnalysis.connect(lambda tabs: self.handle_tabs(tabs))
            except (AttributeError, KeyError):
                pass

    def _init_menu(self):
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.settings_menu = QtWidgets.QMenu('&Settings', self)
        self.settings_menu.addAction('&Global', self.global_settings)
        self.menuBar().addMenu(self.settings_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('&About', self.about)

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    "Version\n%s.\n\n"
                                    "Authors\n%s\n\n"
                                    "GUI authors\n%s" % (testbeam_analysis.VERSION,
                                                         AUTHORS.replace(', ', '\n'),
                                                         GUI_AUTHORS.replace(', ', '\n')))

    def handle_messages(self, message, ms):
        """
        Handles messages from the tabs shown in QMainWindows statusBar
        """

        self.statusBar().showMessage(message, ms)

    def handle_tabs(self, names, enable=True):
        """
        Enables/Disables a specific tab with name 'names' or loops over list of tab names to en/disable them
        """

        if type(names) is unicode:
            # self.tw[names].setDisabled(enable)
            if names in self.tab_order:
                self.tabs.setTabEnabled(self.tab_order.index(names), enable)
        else:
            for name in names:
                # self.tw[name].setDisabled(enable)
                if name in self.tab_order:
                    self.tabs.setTabEnabled(self.tab_order.index(name), enable)

    def update_data(self, data):
        """
        Updates the setup and options with data from the SetupTab

        :param data: dict with all information necessary to perform analysis
        """
        print data
        for key in data:

            if key in self.setup.keys():
                self.setup[key] = data[key]

            elif key in self.options.keys():
                self.options[key] = data[key]

    def update_globals(self):
        """
        Updates the global settings which are applied in the SettingsWindow()
        """

        self.setup = self.sw.setup
        self.options = self.sw.options

        for i in range(self.tabs.count()):
            # FIXME: We need update settings/options method for tabs
            print self.tabs.widget(i)

    def file_quit(self):
        self.close()

    def global_settings(self):
        self.sw = SettingsWindow(self)
        self.sw.show()
        self.sw.settingsUpdated.connect(lambda: self.update_globals())

    def closeEvent(self, _):
        self.file_quit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setPointSize(11)
    app.setFont(font)
    aw = AnalysisWindow()
    aw.show()
    sys.exit(app.exec_())
