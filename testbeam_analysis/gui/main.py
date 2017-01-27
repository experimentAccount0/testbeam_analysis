import sys

from PyQt5 import QtCore, QtWidgets, QtGui
from pyqtgraph.dockarea import DockArea, Dock

from data_table import DataTable, DropArea

PROJECT_NAME = 'Testbeam Analysis'


class AnalysisWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        ''' TODO: DOCSTRING'''
        super(AnalysisWindow, self).__init__(parent)

        self.setWindowTitle(PROJECT_NAME)

        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(0.75 * self.screen.width(), 0.75 * self.screen.height())

        self.init_UI()
        self.init_setup()
        self.file_tab()

    def init_UI(self):
        ''' TODO: DOCSTRING'''
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.menu()

        # Add tabs and widgets for the different analysis steps
        self.tab_order = ('Files', 'Setup', 'HotPixelRemoval', 'Clustering',
                          'Alignment', 'TrackAnalysis')
        self.tab_widgets = {}

        # Add QTabWidget for tabs
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Add DockArea to each tab
        for step in self.tab_order:
            self.tab_widgets[step] = DockArea()
            tabs.addTab(self.tab_widgets[step], step)

        self.statusBar().showMessage(
            "Hello and welcome to a simple and easy to use testbeam analysis!",
            4000)

    def init_setup(self):

        self.input_files = list()
        self.output_folder = str()
        self.z_positions = 0
        self.dut_names = 0
        self.pixel_size = 0
        self.n_pixels = 0

    def update_setup(self):

        self.input_files = self.data_table.update_data()
        self.dut_names = self.data_table.update_dut_names()

    def file_tab(self):

        # Setup widgets
        self.button_size = QtCore.QSize(150, 40)

        self.select_widget = QtWidgets.QWidget()
        self.select_layout = QtWidgets.QGridLayout(self.select_widget)
        select_button = QtGui.QPushButton('Select data of DUTs')
        select_button.setMaximumSize(self.button_size)
        dut_name_button = QtGui.QPushButton('Set DUT names')
        dut_name_button.setMaximumSize(self.button_size)
        da = DropArea()
        self.select_layout.addWidget(da, 0, 0, 1, 1)
        self.select_layout.addWidget(select_button, 0, 1, 1, 1)
        self.select_layout.addWidget(dut_name_button, 0, 2, 1, 1)

        self.handle_widget = QtWidgets.QWidget()
        self.handle_layout = QtWidgets.QGridLayout(self.handle_widget)
        up_button = QtGui.QPushButton('Move up')
        up_button.setMaximumSize(self.button_size)
        down_button = QtGui.QPushButton('Move down')
        down_button.setMaximumSize(self.button_size)
        remove_button = QtGui.QPushButton('Remove DUT')
        remove_button.setMaximumSize(self.button_size)
        self.handle_layout.addWidget(up_button, 0, 0, 1, 1)
        self.handle_layout.addWidget(down_button, 0, 1, 1, 1)
        self.handle_layout.addWidget(remove_button, 0, 2, 1, 1)

        self.table_widget = QtWidgets.QWidget()
        self.table_layout = QtWidgets.QVBoxLayout(self.table_widget)
        self.data_table = DataTable()
        self.table_layout.addWidget(self.data_table)

        # Connect buttons
        # da.fileDrop.connect(self.data_table.get_data(self.table_widget, da.returnData()))
        select_button.clicked.connect(
            lambda: self.data_table.get_data(self.table_widget,
                                             self.input_files))

        for x in [lambda: self.data_table.move_up(),
                  lambda: self.update_setup(),
                  lambda: self.data_table.check_data(self.input_files)]:
            up_button.clicked.connect(x)

        for x in [lambda: self.data_table.move_down(),
                  lambda: self.update_setup(),
                  lambda: self.data_table.check_data(self.input_files)]:
            down_button.clicked.connect(x)

        for x in [lambda: self.data_table.remove_data(),
                  lambda: self.update_setup(),
                  lambda: self.data_table.handle_data(self.table_widget,
                                                      self.input_files)]:
            remove_button.clicked.connect(x)

        for x in [lambda: self.data_table.set_dut_names(),
                  lambda: self.update_setup(),
                  lambda: self.data_table.check_data(self.input_files)]:
            dut_name_button.clicked.connect(x)

        select_dock = Dock('Select DUTs')
        select_dock.setMaximumSize(self.screen.width() / 2, 150)
        handle_dock = Dock('Handle selected \n DUT')
        handle_dock.setMaximumSize(self.screen.width() / 2, 150)
        table_dock = Dock('Selected DUTs')
        select_dock.addWidget(self.select_widget)
        handle_dock.addWidget(self.handle_widget)
        table_dock.addWidget(self.table_widget)
        self.tab_widgets[self.tab_order[0]].addDock(select_dock, 'left')
        self.tab_widgets[self.tab_order[0]].addDock(handle_dock, 'right')
        self.tab_widgets[self.tab_order[0]].addDock(table_dock, 'bottom')

    def menu(self):
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
    aw = AnalysisWindow()
    aw.show()
    sys.exit(app.exec_())
