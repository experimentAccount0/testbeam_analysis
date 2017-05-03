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

# Create all tabs at start up for debugging purpose
_DEBUG = False

try:
    pkgInfo = get_distribution('testbeam_analysis').get_metadata('PKG-INFO')
    AUTHORS = message_from_string(pkgInfo)['Author']
except (DistributionNotFound, KeyError):
    AUTHORS = 'Not defined'


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """
        Initializes the analysis window
        """
        super(AnalysisWindow, self).__init__(parent)

        # Get default settings
        self.setup = SettingsWindow().default_setup
        self.options = SettingsWindow().default_options

        # Make variable for SettingsWindow
        self.settings_window = None

        # Make dict to access tab widgets
        self.tw = {}

        self._init_UI()

    def _init_UI(self):
        """
        Initializes the user interface and displays "Hello"-message
        """

        # Main window settings
        self.setWindowTitle(PROJECT_NAME)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setMinimumSize(MINIMUM_RESOLUTION[0], MINIMUM_RESOLUTION[1])
        self.resize(0.75 * self.screen.width(), 0.75 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Add widgets to main window
        self._init_menu()
        self._init_tabs()
        self.connect_tabs()

        self.handle_messages("Hello and welcome to a simple and easy to use testbeam analysis!", 4000)

    def _init_tabs(self):
        """
        Initializes the tabs for the analysis window
        """

        # Add tab_widget and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'Noisy Pixel', 'Clustering', 'Pre-alignment', 'Track finding',
                          'Alignment', 'Track fitting', 'Analysis', 'Result')

        # Add QTabWidget for tab_widget
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize each tab
        for name in self.tab_order:
            if name == 'Files':
                widget = DataTab(parent=self.tabs)
            else:
                # Add dummy widget
                widget = QtWidgets.QWidget(parent=self.tabs)

            self.tw[name] = widget
            self.tabs.addTab(self.tw[name], name)

        # Disable all tabs but DataTab. Enable tabs later via self.enable_tabs()
        if not _DEBUG:
            self.handle_tabs(enable=False)
        else:
            self.handle_tabs(enable=True)
            #self.update_tabs()

    def _init_menu(self):
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.settings_menu = QtWidgets.QMenu('&Settings', self)
        self.settings_menu.addAction('&Global', self.global_settings)
        self.menuBar().addMenu(self.settings_menu)

        self.session_menu = QtWidgets.QMenu('&Session', self)
        self.session_menu.addAction('&Save', self.save_session, QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        self.session_menu.addAction('&Load', self.load_session, QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.menuBar().addMenu(self.session_menu)

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

    def handle_tabs(self, tabs=None, enable=True):
        """
        Enables/Disables a specific tab with name 'names' or loops over list of tab names to en/disable them
        """

        if _DEBUG:
            return
        # Dis/enable all tabs but Files
        if tabs is None:
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) != 'Files':
                    self.tabs.setTabEnabled(i, enable)

        # Dis/enable specific tab
        elif type(tabs) is unicode or type(tabs) is str:
            if tabs in self.tab_order:
                self.tabs.setTabEnabled(self.tab_order.index(tabs), enable)

        # Dis/enable several tabs
        else:
            for tab in tabs:
                if tab in self.tab_order:
                    self.tabs.setTabEnabled(self.tab_order.index(tab), enable)

    def connect_tabs(self, tabs=None):
        """
        Connect statusMessage and proceedAnalysis signal of all tabs
        """

        if tabs is None:
            tab_list = self.tab_order
        else:
            tab_list = [tabs]

        for name in tab_list:
            try:
                if name == 'Files':
                    self.tw[name].proceedAnalysis.connect(lambda: self.update_tabs(tabs='Setup'))
                    self.tw[name].proceedAnalysis.connect(lambda: self.tw['Setup'].input_data(self.tw['Files'].data))

                if name == 'Setup':
                    self.tw[name].proceedAnalysis.connect(lambda: self.update_tabs(data=self.tw['Setup'].data,
                                                                                   skip='Setup'))

                self.tw[name].statusMessage.connect(lambda message: self.handle_messages(message, 4000))
                self.tw[name].proceedAnalysis.connect(lambda tab_names: self.handle_tabs(tabs=tab_names))
#                self.tw[name].proceedAnalysis.connect(lambda: self.tabs.setCurrentIndex(self.tabs.currentIndex()+1))

            except (AttributeError, KeyError):
                print name, AttributeError, KeyError
                pass

    def update_tabs(self, data=None, tabs=None, skip=None):
        """
        Updates the setup and options with data from the SetupTab and then updates the tabs

        :param tabs: list of strings with tab names that should be updated, if None update all
        :param data: dict with all information necessary to perform analysis, if None only update tabs
        :param skip: str or list of tab names which should be skipped when updating tabs
        """

        # Save users current tab position
        current_tab = self.tabs.currentIndex()

        if data is not None:

            for key in data:

                # Store setup data in self.setup and everything else in self.options
                if key in self.setup.keys():
                    self.setup[key] = data[key]
                else:
                    self.options[key] = data[key]

        if tabs is None:
            update_tabs = list(self.tab_order)
        else:
            update_tabs = [tabs]

        if skip is not None:
            if isinstance(skip, list):
                for tab in skip:
                    if tab in update_tabs:
                        update_tabs.remove(tab)
            else:
                update_tabs.remove(skip)

        # Make temporary dict for updated tabs
        tmp_tw = {}
        for name in update_tabs:

            if name == 'Setup':
                widget = SetupTab(parent=self.tabs)

            elif name == 'Noisy Pixel':
                widget = tab_widget.NoisyPixelsTab(parent=self.tabs,
                                                   setup=self.setup,
                                                   options=self.options)
            elif name == 'Clustering':
                widget = tab_widget.ClusterPixelsTab(parent=self.tabs,
                                                     setup=self.setup,
                                                     options=self.options,
                                                     tab_list='Pre-alignment')

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
                logging.info('Gui for %s not implemented yet!' % name)
                continue

            tmp_tw[name] = widget

        for tab in self.tab_order:
            if tab in tmp_tw.keys():

                # Replace tabs in self.tw with updated tabs
                self.tw[tab] = tmp_tw[tab]

                # Get tab status of tab which is updated to set status of updated tab
                enable = self.tabs.isTabEnabled(self.tab_order.index(tab))

                # Remove old tab, insert updated tab at same index and set status
                self.tabs.removeTab(self.tab_order.index(tab))
                self.tabs.insertTab(self.tab_order.index(tab), self.tw[tab], tab)
                self.tabs.setTabEnabled(self.tab_order.index(tab), enable)

        # Set the tab index to stay at the same tab after replacing old tabs
        self.tabs.setCurrentIndex(current_tab)

        # Connect updated tabs
        self.connect_tabs(tabs)

    def global_settings(self):
        """
        Creates a child SettingsWindow of the analysis window to change global settings
        """
        self.settings_window = SettingsWindow(self.setup, self.options, parent=self)
        self.settings_window.show()
        self.settings_window.settingsUpdated.connect(lambda: self.update_globals())

    def update_globals(self):
        """
        Updates the global settings which are applied in the SettingsWindow
        """

        self.setup = self.settings_window.setup
        self.options = self.settings_window.options

        self.update_tabs(skip='Setup')

    def save_session(self):
        """
        Creates a child SessionWindow of the analysis window to save current session
        """
        caption = 'Save session'
        session = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                        caption=caption,
                                                        directory='./sessions',
                                                        filter='*.yaml')[0]
        pass

    def load_session(self):
        """
        Opens dialog to select previously saved session. Must be yaml-file and lie in ./sessions
        """
        caption = 'Load session'
        session = QtWidgets.QFileDialog.getOpenFileName(parent=self,
                                                        caption=caption,
                                                        directory='./sessions',
                                                        filter='*.yaml')[0]
        pass

    def file_quit(self):
        self.close()

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
