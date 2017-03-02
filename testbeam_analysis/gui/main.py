import sys
import logging

from PyQt5 import QtCore, QtWidgets, QtGui

from data import DataTab
from settings import SettingsWindow, DefaultSettings
from testbeam_analysis.gui import tab_widget

PROJECT_NAME = 'Testbeam Analysis'


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """ TODO: DOCSTRING"""
        super(AnalysisWindow, self).__init__(parent)
        self.setup = DefaultSettings().setup
        self.options = DefaultSettings().options
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

        self.statusBar().showMessage("Hello and welcome to a simple and easy to use testbeam analysis!", 4000)

    def _init_tabs(self):
        # Add tab_widget and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'Noisy Pixel', 'Clustering',
                          'Correlations', 'Pre-alignment', 'Track finding',
                          'Alignment', 'Track fitting', 'Analysis')

        # Add QTabWidget for tab_widget
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Initialize each tab
        for name in self.tab_order:
            if name == 'Files':
                widget = DataTab(parent=tabs, parent_window=self)
            elif name == 'Noisy Pixel':
                widget = tab_widget.NoisyPixelsTab(parent=tabs,
                                                   setup=self.setup,
                                                   options=self.options)
            elif name == 'Clustering':
                widget = tab_widget.ClusterPixelsTab(parent=tabs,
                                                     setup=self.setup,
                                                     options=self.options)
            elif name == 'Correlations':
                widget = tab_widget.CorrelateClusterTab(parent=tabs,
                                                        setup=self.setup,
                                                        options=self.options)
            elif name == 'Pre-alignment':
                widget = tab_widget.PrealignmentTab(parent=tabs,
                                                    setup=self.setup,
                                                    options=self.options)
            elif name == 'Track finding':
                widget = tab_widget.TrackFindingTab(parent=tabs,
                                                    setup=self.setup,
                                                    options=self.options)
            elif name == 'Alignment':
                widget = tab_widget.AlignmentTab(parent=tabs,
                                                 setup=self.setup,
                                                 options=self.options)
            elif name == 'Track fitting':
                widget = tab_widget.TrackFittingTab(parent=tabs,
                                                    setup=self.setup,
                                                    options=self.options)
            else:
                logging.info('GUI for %s not implemented yet', name)
                continue
            tabs.addTab(widget, name)

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

    def file_quit(self):
        self.close()

    def global_settings(self):
        sw = SettingsWindow(self)
        sw.show()

    def closeEvent(self, _):
        self.file_quit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setFamily(font.defaultFamily())
    font.setPointSize(11)
    app.setFont(font)
    aw = AnalysisWindow()
    aw.show()
    sys.exit(app.exec_())
