import sys
import logging

from PyQt5 import QtCore, QtWidgets, QtGui
from pyqtgraph.dockarea import DockArea

from data import DataTab
from testbeam_analysis.gui import tab_widget

PROJECT_NAME = 'Testbeam Analysis'


# For testing until widget provide this info
setup = {'n_pixel': (10, 10),
         'pixel_size': (100, 100),
         'dut_name': 'myDUT'}

options = {'working_directory': '',
           'input_hits_file': 'test_DUT0.h5',
           'output_mask_file': 'tt',
           'chunk_size': 1000000,
           'plot': False}


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        ''' TODO: DOCSTRING'''
        super(AnalysisWindow, self).__init__(parent)

        self._init_UI()

    def _init_UI(self):
        ''' TODO: DOCSTRING'''

        # Main window settings
        self.setWindowTitle(PROJECT_NAME)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(0.75 * self.screen.width(), 0.75 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Add widgets to main window
        self._init_menu()
        self._init_tabs()

        self.statusBar().showMessage(
            "Hello and welcome to a simple and easy to use testbeam analysis!",
            4000)

    def _init_tabs(self):
        # Add tab_widget and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'Noisy Pixel', 'Clustering',
                          'Correlations', 'Pre-alignment', 'Track finding', 'Alignment',
                          'Track fitting', 'Track Analysis')

        # Add QTabWidget for tab_widget
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Initialize each tab
        for name in self.tab_order:
            if name == 'Files':
                continue
                widget = DockArea()
                self.data_tab = DataTab(parent=widget)
            elif name == 'Noisy Pixel':
                widget = tab_widget.NoisyPixelsTab(parent=tabs,
                                                   setup=setup,
                                                   options=options)
            elif name == 'Clustering':
                widget = tab_widget.ClusterPixelsTab(parent=tabs,
                                                     setup=setup,
                                                     options=options)
            elif name == 'Correlations':
                widget = tab_widget.CorrelateClusterTab(parent=tabs,
                                                        setup=setup,
                                                        options=options)
            else:
                #                 logging.warning('GUI for %s not implemented yet', name)
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

        # self.help_menu.addAction('&About', self.about)

    def file_quit(self):
        self.close()

    def closeEvent(self, _):
        self.file_quit()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    font = QtGui.QFont()
    font.setPointSize(12)
    app.setFont(font)
    aw = AnalysisWindow()
    aw.show()
    sys.exit(app.exec_())
