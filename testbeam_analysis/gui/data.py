import tables as tb

from pyqtgraph.dockarea import Dock
from PyQt5 import QtCore, QtWidgets, QtGui


class DataTab(object):
    ''' Implements the tab content for data file handling'''

    def __init__(self, parent):
        self.parent = parent
        self.input_files = list()
        self.output_folder = str()

        self._setup()
#         self._setup_test()
#
#     def _setup_test(self):
#         self.items = QtWidgets.QDockWidget("Dockable", self)
#         pass

    def _setup(self):
        # Setup widgets
        button_size = QtCore.QSize(150, 40)

        select_widget = QtWidgets.QWidget()
        select_layout = QtWidgets.QGridLayout(select_widget)
        select_button = QtGui.QPushButton('Select data of DUTs')
        select_button.setMaximumSize(button_size)
        dut_name_button = QtGui.QPushButton('Set DUT names')
        dut_name_button.setMaximumSize(button_size)
        da = DropArea()
        select_layout.addWidget(da, 0, 0, 1, 1)
        select_layout.addWidget(select_button, 0, 1, 1, 1)
        select_layout.addWidget(dut_name_button, 0, 2, 1, 1)

        handle_widget = QtWidgets.QWidget()
        handle_layout = QtWidgets.QGridLayout(handle_widget)
        up_button = QtGui.QPushButton('Move up')
        up_button.setMaximumSize(button_size)
        down_button = QtGui.QPushButton('Move down')
        down_button.setMaximumSize(button_size)
        remove_button = QtGui.QPushButton('Remove DUT')
        remove_button.setMaximumSize(button_size)
        handle_layout.addWidget(up_button, 0, 0, 1, 1)
        handle_layout.addWidget(down_button, 0, 1, 1, 1)
        handle_layout.addWidget(remove_button, 0, 2, 1, 1)

        table_widget = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_widget)
        self.data_table = DataTable(self.parent)
        table_layout.addWidget(self.data_table)

        # Connect buttons
        # da.fileDrop.connect(data_table.get_data(table_widget, da.returnData()))
        select_button.clicked.connect(
            lambda: self.data_table.get_data(table_widget,
                                             self.input_files))

        for x in [lambda: self.data_table.move_up(),
                  lambda: self._update_setup(),
                  lambda: self.data_table.check_data(self.input_files)]:
            up_button.clicked.connect(x)

        for x in [lambda: self.data_table.move_down(),
                  lambda: self._update_setup(),
                  lambda: self.data_table.check_data(self.input_files)]:
            down_button.clicked.connect(x)

        for x in [lambda: self.data_table.remove_data(),
                  lambda: self._update_setup(),
                  lambda: self.data_table.handle_data(table_widget,
                                                      self.input_files)]:
            remove_button.clicked.connect(x)

        for x in [lambda: self.data_table.set_dut_names(),
                  lambda: self._update_setup(),
                  lambda: self.data_table.check_data(self.input_files)]:
            dut_name_button.clicked.connect(x)

        select_dock = Dock('Select DUTs')
#         select_dock.setMaximumSize(self.screen.width() / 2, 150)
        handle_dock = Dock('Handle selected \n DUT')
#         handle_dock.setMaximumSize(self.screen.width() / 2, 150)
        table_dock = Dock('Selected DUTs')
        select_dock.addWidget(select_widget)
        handle_dock.addWidget(handle_widget)
        table_dock.addWidget(table_widget)
        self.parent.addDock(select_dock, 'left')
        self.parent.addDock(handle_dock, 'right')
        self.parent.addDock(table_dock, 'bottom')

    def _update_setup(self):
        self.input_files = self.data_table.update_data()
        self.dut_names = self.data_table.update_dut_names()


class DataTable(QtWidgets.QTableWidget):
    ''' TODO: DOCSTRING MISSING'''

    def __init__(self, parent=None):
        super(DataTable, self).__init__(parent)

        self.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.showGrid()
        self.setSortingEnabled(True)

    def move_down(self):
        ''' Move current row down one place '''
        row = self.currentRow()
        column = self.currentColumn()
        if row < self.rowCount() - 1:
            self.insertRow(row + 2)
            for i in range(self.columnCount()):
                self.setItem(row + 2, i, self.takeItem(row, i))
                self.setCurrentCell(row + 2, column)
            self.removeRow(row)
            self.setVerticalHeaderLabels(self.row_labels)

    def move_up(self):
        ''' Move current row up one place '''
        row = self.currentRow()
        column = self.currentColumn()
        if row > 0:
            self.insertRow(row - 1)
            for i in range(self.columnCount()):
                self.setItem(row - 1, i, self.takeItem(row + 1, i))
                self.setCurrentCell(row - 1, column)
            self.removeRow(row + 1)
            self.setVerticalHeaderLabels(self.row_labels)

    def set_dut_names(self, name='Tel'):
        ''' Set DUT names for further analysis

            Std. setting is Tel_i  and i is index
        '''
        for i in range(self.rowCount()):
            dutItem = QtGui.QTableWidgetItem()
            dutItem.setTextAlignment(QtCore.Qt.AlignCenter)
            dutItem.setText(name + '_%d' % i)
            self.setItem(i, 1, dutItem)

    def update_dut_names(self):
        ''' Read list of DUT names from table and return tuple of names'''

        new = []
        try:
            for row in range(self.rowCount()):
                new.append(self.item(row, 1).text())
        except AttributeError:
            return None

        return tuple(new)

    def get_data(self, table_widget, input_files):
        ''' Open file dialog and select data files

            Only *.h5 files are allowed '''

        caption = 'Select data of DUTs'
        for path in QtGui.QFileDialog.getOpenFileNames(self,
                                                       caption=caption,
                                                       directory='~/',
                                                       filter='*.h5')[0]:
            input_files.append(path)

        self.handle_data(table_widget, input_files)

    def handle_data(self, table_widget, input_files):
        ''' Arranges input_data in the table and re-news table if DUT amount/order has been updated '''

        for widget in table_widget.children():
            if isinstance(widget, QtGui.QTableWidget):
                table_widget.layout().removeWidget(widget)

        self.row_labels = [('DUT ' + '%d' % i)
                           for i in range(len(input_files))]
        self.column_labels = ['Path', 'DUT names', 'Status']
        self.setColumnCount(len(self.column_labels))
        self.setRowCount(len(self.row_labels))
        self.setHorizontalHeaderLabels(self.column_labels)
        self.setVerticalHeaderLabels(self.row_labels)

        for i, dut in enumerate(input_files):
            pathItem = QtGui.QTableWidgetItem()
            pathItem.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            pathItem.setTextAlignment(QtCore.Qt.AlignLeft)
            pathItem.setText(dut)
            self.setItem(i, 0, pathItem)

        self.check_data(input_files)

    def check_data(self, input_files):
        ''' Checks if given input_files contain the necessary information like
        event_number, column, row, etc.; visualzes broken input '''

        field_req = ('event_number', 'frame', 'column', 'row', 'charge')
        incompatible_data = {}  # store indices and status of incompatible data

        for i, path in enumerate(input_files):
            with tb.open_file(path, mode='r') as in_file:
                try:
                    if not all(name in field_req for name
                               in in_file.root.Hits.colnames):
                        incompatible_data[i] = 'Data does not contain all ' \
                            'required information!'
                except tb.exceptions.NoSuchNodeError:
                    incompatible_data[i] = 'NoSuchNodeError: Data does not ' \
                        'contain hits!'

        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)

        for row in range(self.rowCount()):
            statusItem = QtGui.QTableWidgetItem()
            statusItem.setFlags(
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            statusItem.setTextAlignment(QtCore.Qt.AlignCenter)

            if row in incompatible_data.keys():
                statusItem.setText(incompatible_data[row])
                self.setItem(row, 2, statusItem)

                for col in range(self.columnCount()):
                    try:
                        self.item(row, col).setFont(font)
                        self.item(row, col).setForeground(QtGui.QColor('red'))
                    except AttributeError:
                        pass
            else:
                statusItem.setText('Okay')
                self.setItem(row, 2, statusItem)

    def update_data(self):
        ''' Updates the data/DUT content/order by re-reading the filespaths
        from the table and returnig a list with the new DUT order '''

        new = []
        for row in range(self.rowCount()):
            new.append(self.item(row, 0).text())
        return new

    def remove_data(self):
        ''' Removes an entire row from the data table e.g. a DUT '''

        row = self.currentRow()
        self.removeRow(row)


class DropArea(QtWidgets.QFrame):

    #fileDrop = QtCore.pyqtSignal()

    def __init__(self):

        QtWidgets.QFrame.__init__(self)
        self.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        self.setFrameShadow(QtGui.QFrame.Sunken)
        self.setLineWidth(3)
        self.setMinimumSize(200, 100)
        self.setMaximumSize(300, 175)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.input_files = []
        self.init_UI()

    def init_UI(self):

        drop_label = QtGui.QLabel('Drag and Drop')
        self.layout.addWidget(drop_label)

    def dragEnterEvent(self, e):

        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):

        if e.mimeData().hasUrls:
            e.setDropAction(QtCore.Qt.CopyAction)
            e.accept()

            for url in e.mimeData().urls():
                self.input_files.append(str(url.toLocalFile()))

        else:
            e.ignore()

        # if len(self.input_files) != 0:
            # self.fileDrop.emit()

    # def returnData(self):
        # return self.input_files
