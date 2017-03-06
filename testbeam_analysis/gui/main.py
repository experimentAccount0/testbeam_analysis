import sys
import logging

from PyQt5 import QtCore, QtWidgets, QtGui

from data import DataTab
from setup import SetupTab
from settings import SettingsWindow, DefaultSettings
from testbeam_analysis.gui import tab_widget

PROJECT_NAME = 'Testbeam Analysis'


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """ TODO: DOCSTRING"""
        super(AnalysisWindow, self).__init__(parent)

        # Get default settings
        self.setup = DefaultSettings().setup
        self.options = DefaultSettings().options

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
                          'Alignment', 'Track fitting', 'Analysis')

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
            elif name == 'Correlations':
                widget = tab_widget.CorrelateClusterTab(parent=self.tabs,
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

        # Connect statusMessage signal of all tabs
        for name in self.tab_order:
            try:
                self.tw[name].statusMessage.connect(lambda message: self.handle_messages(message, 4000))
            except (AttributeError, KeyError):
                pass

        # Connect DataTab
        self.tw['Files'].proceedAnalysis.connect(lambda: self.tw['Setup'].input_data(self.tw['Files'].data))
        self.tw['Files'].proceedAnalysis.connect(lambda: self.handle_tabs('Setup'))

    def _init_menu(self):
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.settings_menu = QtWidgets.QMenu('&Settings', self)
        self.settings_menu.addAction('&Global', self.global_settings)
        self.menuBar().addMenu(self.settings_menu)

        # self.help_menu.addAction('&About', self.about)

    def handle_messages(self, message, ms):
        """
        Handles messages from the tabs shown in QMainWindows statusBar
        """

        self.statusBar().showMessage(message, ms)

    def handle_tabs(self, names, enable=True):
        """
        Enables/Disables a specific tab with name 'names' or loops over list of tab names to en/disable them
        """

        if type(names) is str:
            # self.tw[names].setDisabled(enable)
            self.tabs.setTabEnabled(self.tab_order.index(names), enable)
        else:
            for name in names:
                # self.tw[name].setDisabled(enable)
                self.tabs.setTabEnabled(self.tab_order.index(name), enable)

    def update_settings(self):
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
        self.sw.settingsUpdated.connect(lambda: self.update_settings())

    def closeEvent(self, _):
        self.file_quit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
#    font.setFamily(font.defaultFamily())
    font.setPointSize(11)
    app.setFont(font)
    aw = AnalysisWindow()
    aw.show()
    sys.exit(app.exec_())
