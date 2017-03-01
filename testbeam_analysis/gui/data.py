import tables as tb

from PyQt5 import QtCore, QtWidgets, QtGui


class DataTab(QtWidgets.QWidget):
    """
    Implements the tab content for data file handling
    """

    def __init__(self, parent=None, parent_window=None):
        super(DataTab, self).__init__(parent)

        # Store parent QMainWindow of DataTab
        self.parent_window = parent_window

        self._setup()

    def _setup(self):
        # Table area
        left_widget = QtWidgets.QWidget()
        tab_layout = QtWidgets.QHBoxLayout()
        left_widget.setLayout(tab_layout)
        self.data_table = DataTable(parent=left_widget, parent_window=self.parent_window)
        tab_layout.addWidget(self.data_table)
        # Make option area
        layout_options = QtWidgets.QVBoxLayout()
        layout_options.setSpacing(30)
        label_option = QtWidgets.QLabel('Options')
        layout_options.addWidget(label_option)
        # Make buttons and layout for option buttons
        layout_buttons = QtWidgets.QVBoxLayout()
        layout_buttons.setSpacing(15)
        # Make a select button to select input files
        layout_select = QtWidgets.QHBoxLayout()
        layout_select.addSpacing(10)
        button_select = QtWidgets.QPushButton('Select data of DUTs')
        layout_select.addWidget(button_select)
        button_select.clicked.connect(lambda: self.data_table.get_data())
        # Make button to reset dut names
        layout_names = QtWidgets.QHBoxLayout()
        layout_names.addSpacing(10)
        button_names = QtWidgets.QPushButton('Set DUT names')
        button_names.setToolTip('Set default DUT names')
        layout_names.addWidget(button_names)
        button_names.clicked.connect(lambda: self.data_table.set_dut_names())
        # Make button to clear the table content
        layout_clear = QtWidgets.QHBoxLayout()
        layout_clear.addSpacing(10)
        button_clear = QtWidgets.QPushButton('Clear')
        button_clear.setToolTip('Clears table')
        layout_clear.addWidget(button_clear)
        button_clear.clicked.connect(lambda: self.data_table.clear_table())
        # Make button to select output folder
        layout_output1 = QtWidgets.QHBoxLayout()
        label_output = QtWidgets.QLabel('Output folder')
        checkbox_output = QtWidgets.QCheckBox()
        layout_output1.addWidget(label_output)
        layout_output1.addStretch(0)
        layout_output1.addWidget(checkbox_output)
        layout_output2 = QtWidgets.QHBoxLayout()
        layout_output2.addSpacing(10)
        button_output = QtWidgets.QPushButton()
        icon_output = button_output.style().standardIcon(QtWidgets.QStyle.SP_FileDialogStart)
        button_output.setIcon(icon_output)
        button_output.setIconSize(QtCore.QSize(30, 30))
        button_output.setFixedSize(QtCore.QSize(35, 35))
        button_output.setToolTip('Set output older')
        edit_output = QtWidgets.QLineEdit()
        edit_output.setFixedHeight(35)
        layout_output2.addWidget(button_output)
        layout_output2.addWidget(edit_output)
        # Add to main layout
        layout_buttons.addLayout(layout_select)
        layout_buttons.addLayout(layout_names)
        layout_buttons.addLayout(layout_clear)
        layout_options.addLayout(layout_buttons)
        layout_options.addLayout(layout_output1)
        layout_options.addLayout(layout_output2)
        # Make proceed button
        self.button_ok = QtWidgets.QPushButton('Ok')
        layout_options.addStretch(0)
        layout_options.addWidget(self.button_ok)
        self.button_ok.setDisabled(True)
        self.button_ok.setToolTip('Select data of DUTs')
        self.button_ok.clicked.connect(lambda: self.data_table.save_config())
        self.data_table.inputFilesChanged.connect(lambda: self.analysis_check())
        # Add main layout to widget
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(layout_options)
        # Split table and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 2.5)
        widget_splitter.setChildrenCollapsible(False)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def analysis_check(self):
        """
        Handles  whether the proceed 'OK' button is clickable or not in regard to the input data.
        If not, respective messages are shown in QMainWindows statusBar
        """
        if len(self.data_table.input_files) > 0 and len(self.data_table.incompatible_data) == 0:
            self.button_ok.setDisabled(False)
            self.button_ok.setToolTip('Proceed')

        else:
            self.button_ok.setDisabled(True)
            self.button_ok.setToolTip('Select data of DUTs')

            if self.parent_window is not None:
                if len(self.data_table.incompatible_data) != 0:
                    broken = []
                    for key in self.data_table.incompatible_data.keys():
                        broken.append(self.data_table.dut_names[key])

                    self.parent_window.statusBar().showMessage("Data of %s is broken. Analysis impossible." %
                                                               str(',').join(broken), 4000)

                if len(self.data_table.input_files) == 0:
                    self.parent_window.statusBar().showMessage("No data. Analysis impossible.", 4000)


class DataTable(QtWidgets.QTableWidget):
    """
    Class to get, display and handle the input data of the DUTs
    for which a testbeam analysis will be performed
    """

    inputFilesChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None, parent_window=None):
        super(DataTable, self).__init__(parent)

        # Store parent QMainWindow of DataTable
        self.parent_window = parent_window

        # Dict for returning
        self.dut_data = {'input_files': None, 'dut_names': None}

        # Lists for dut names, input files, etc.
        self.dut_names = []
        self.input_files = []

        # store indices and status of incompatible data might occurring in check_data
        self.incompatible_data = {}

        # Appearance
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.setWordWrap(True)
        self.setTextElideMode(QtCore.Qt.ElideLeft)
        self.showGrid()
        self.setSortingEnabled(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

    def get_data(self):
        """
        Open file dialog and select data files. Only *.h5 files are allowed
        """

        caption = 'Select data of DUTs'
        for path in QtWidgets.QFileDialog.getOpenFileNames(parent=self,
                                                           caption=caption,
                                                           directory='~/',
                                                           filter='*.h5')[0]:
            self.input_files.append(path)

        self.handle_data()

    def handle_data(self):
        """
        Arranges input_data in the table and re-news table if DUT amount/order has been updated
        """

#        for widget in self.parentWidget().children():
#            if isinstance(widget, QtWidgets.QTableWidget):
#                #widget.clear()
#                self.parentWidget().layout().removeWidget(widget)
#        self.clear()

        self.row_labels = [('DUT ' + '%d' % i) for i, _ in enumerate(self.input_files)]
        self.column_labels = ['Path', 'DUT name', 'Status', 'Navigation']

        self.setColumnCount(len(self.column_labels))
        self.setRowCount(len(self.row_labels))
        self.setHorizontalHeaderLabels(self.column_labels)
        self.setVerticalHeaderLabels(self.row_labels)

        for row, dut in enumerate(self.input_files):
            path_item = QtWidgets.QTableWidgetItem()
            path_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            path_item.setTextAlignment(QtCore.Qt.AlignLeft)
            path_item.setText(dut)
            self.setItem(row, self.column_labels.index('Path'), path_item)

        self.update_dut_names()
        self.check_data()
        self._make_nav_buttons()
        self.inputFilesChanged.emit()

    def check_data(self):
        """
        Checks if given input_files contain the necessary information like
        event_number, column, row, etc.; visualizes broken input
        """

        field_req = ('event_number', 'frame', 'column', 'row', 'charge')
        self.incompatible_data = dict()

        for i, path in enumerate(self.input_files):
            with tb.open_file(path, mode='r') as f:
                try:
                    fields = f.root.Hits.colnames
                    missing = []
                    for req in field_req:
                        if req in fields:
                            pass
                        else:
                            missing.append(req)
                    if len(missing) != 0:
                        self.incompatible_data[i] = 'Error! Data does not contain field(s):\n' + ', '.join(missing)

                except tb.exceptions.NoSuchNodeError:
                    self.incompatible_data[i] = 'NoSuchNodeError: Nodes:\n' + str(f.root)

        font = QtGui.QFont()
        font.setBold(True)

        for row in range(self.rowCount()):
            status_item = QtWidgets.QTableWidgetItem()
            status_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)

            if row in self.incompatible_data.keys():
                error_font = font
                error_font.setUnderline(True)
                status_item.setText(self.incompatible_data[row])
                self.setItem(row, self.column_labels.index('Status'), status_item)

                for col in range(self.columnCount()):
                    try:
                        self.item(row, col).setFont(error_font)
                        self.item(row, col).setForeground(QtGui.QColor('red'))
                    except AttributeError:
                        pass
            else:
                status_item.setText('Okay')
                self.setItem(row, self.column_labels.index('Status'), status_item)
                self.item(row, self.column_labels.index('Status')).setFont(font)
                self.item(row, self.column_labels.index('Status')).setForeground(QtGui.QColor('green'))

    def update_data(self):
        """
        Updates the data/DUT content/order by re-reading the filespaths
        from the table and updating self.input_files
        """

        new = []
        try:
            for row in range(self.rowCount()):
                new.append(self.item(row, self.column_labels.index('Path')).text())
        except AttributeError:
            pass

        if new != self.input_files:  # and len(new) != 0:
            self.input_files = new
            self.inputFilesChanged.emit()

    def set_dut_names(self, name='Tel'):
        """
        Set DUT names for further analysis. Std. setting is Tel_i  and i is index
        """

        for row in range(self.rowCount()):
            dut_name = name + '_%d' % row
            dut_item = QtWidgets.QTableWidgetItem()
            dut_item.setTextAlignment(QtCore.Qt.AlignCenter)
            dut_item.setText(dut_name)
            if row in self.incompatible_data.keys():
                font = dut_item.font()
                font.setBold(True)
                font.setUnderline(True)
                dut_item.setFont(font)
                dut_item.setForeground(QtGui.QColor('red'))
            self.setItem(row, self.column_labels.index('DUT name'), dut_item)
            self.dut_names.append(dut_name)

    def update_dut_names(self, name='Tel'):
        """
        Read list of DUT names from table and update dut names. Also add new DUT names and update
        """

        new = []
        try:
            for row in range(self.rowCount()):
                try:
                    new.append(self.item(row, self.column_labels.index('DUT name')).text())
                except AttributeError: # no QTableWidgetItem for new input data
                    add_dut_item = QtWidgets.QTableWidgetItem()
                    add_dut_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    add_dut_item.setText(name + '_%d' % row)
                    self.setItem(row, self.column_labels.index('DUT name'), add_dut_item)
                    new.append(self.item(row, self.column_labels.index('DUT name')).text())
        except AttributeError: # no QTableWidgetItem has been created at all
            self.set_dut_names()

        if new != self.dut_names:  # and len(new) != 0:
            self.dut_names = new
            for row in range(self.rowCount()):
                self.item(row, self.column_labels.index('DUT name')).setText(self.dut_names[row])

#        self.resizeRowsToContents()
#        self.resizeColumnsToContents()

    def update_setup(self):
        """
        Updating all relevant lists for further analysis
        """

        self.update_data()
        self.handle_data()
#        self.update_dut_names()

    def save_config(self):
        """
        Save configuration from table and get parent QMainWindow and ShowMessage
        """
        self.update_setup()
        self.dut_data['input_files'] = self.input_files
        self.dut_data['dut_names'] = self.dut_names

        if self.parent_window is not None:
            self.parent_window.statusBar().showMessage("Configuration for %d DUT(s) saved" %
                                                       (len(self.input_files)), 2000)

    def clear_table(self):
        """
        Clear table of all its contents
        """

        self.setRowCount(0)
        self.update_data()

    def _make_nav_buttons(self):
        """
        Make buttons to navigate through table and delete entries
        """

        for row in range(self.rowCount()):
            widget_but = QtWidgets.QWidget()
            layout_but = QtWidgets.QHBoxLayout()
            layout_but.setAlignment(QtCore.Qt.AlignCenter)
            self.button_up = QtWidgets.QPushButton()
            self.button_down = QtWidgets.QPushButton()
            self.button_del = QtWidgets.QPushButton()
            button_size = QtCore.QSize(40,40)
            icon_up = self.button_up.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp)
            icon_down = self.button_down.style().standardIcon(QtWidgets.QStyle.SP_ArrowDown)
            icon_del = self.button_del.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
            icon_size = QtCore.QSize(30, 30)
            self.button_up.setIcon(icon_up)
            self.button_down.setIcon(icon_down)
            self.button_del.setIcon(icon_del)
            self.button_up.setIconSize(icon_size)
            self.button_down.setIconSize(icon_size)
            self.button_del.setIconSize(icon_size)
            self.button_up.setFixedSize(button_size)
            self.button_down.setFixedSize(button_size)
            self.button_del.setFixedSize(button_size)
            self.button_del.setToolTip('Delete')
            self.button_up.setToolTip('Move up')
            self.button_down.setToolTip('Move down')

            for x in [lambda: self._move_up(), lambda: self.update_setup()]:
                self.button_up.clicked.connect(x)

            for x in [lambda: self._move_down(), lambda: self.update_setup()]:
                self.button_down.clicked.connect(x)

            for x in [lambda: self._delete_data(), lambda: self.handle_data()]:
                self.button_del.clicked.connect(x)

            layout_but.addWidget(self.button_up)
            layout_but.addWidget(self.button_down)
            layout_but.addWidget(self.button_del)
            widget_but.setLayout(layout_but)
            self.setCellWidget(row, self.column_labels.index('Navigation'), widget_but)

    def _delete_data(self):
        """
        Deletes row at sending button position
        """

        button = self.sender()
        index = self.indexAt(button.parentWidget().pos())
        if index.isValid():
            row = index.row()
            self.removeRow(row)
            self.input_files.pop(row)
            if row in self.incompatible_data.keys():
                self.incompatible_data.pop(row)
            self.inputFilesChanged.emit()

    def _move_down(self):
        """
        Move row at sending button position one place down
        """

        button= self.sender()
        index = self.indexAt(button.parentWidget().pos())
        row = index.row()
        column = index.column()
        if row < self.rowCount() - 1:
            self.insertRow(row + 2)
            for i in range(self.columnCount()):
                self.setItem(row + 2, i, self.takeItem(row, i))
                self.setCurrentCell(row + 2, column)
            self.removeRow(row)
            self.setVerticalHeaderLabels(self.row_labels)

    def _move_up(self):
        """
        Move row at sending button position one place up
        """

        button = self.sender()
        index = self.indexAt(button.parentWidget().pos())
        row = index.row()
        column = index.column()
        if row > 0:
            self.insertRow(row - 1)
            for i in range(self.columnCount()):
                self.setItem(row - 1, i, self.takeItem(row + 1, i))
                self.setCurrentCell(row - 1, column)
            self.removeRow(row + 1)
            self.setVerticalHeaderLabels(self.row_labels)
