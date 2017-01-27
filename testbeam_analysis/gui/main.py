import sys

from PyQt5 import QtCore, QtWidgets, QtGui
from pyqtgraph.dockarea import DockArea, Dock

from data_tab import DataTable, DropArea, DataTab

PROJECT_NAME = 'Testbeam Analysis'


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
        # Add tabs and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'Noisy Pixel', 'Clustering',
                          'Correlations', 'Pre-alignment', 'Track finding', 'Alignment',
                          'Track fitting', 'Track Analysis')
        self.tab_widgets = {}

        # Add QTabWidget for tabs
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Add DockArea to each tab
        for name in self.tab_order:
            self.tab_widgets[name] = DockArea()
            tabs.addTab(self.tab_widgets[name], name)

        # Init tab number 1 with data
        self.data_tab = DataTab(parent=self.tab_widgets[self.tab_order[0]])

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
