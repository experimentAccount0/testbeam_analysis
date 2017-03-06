import logging
import json
from PyQt5 import QtCore, QtWidgets, QtGui


class SetupTab(QtWidgets.QWidget):
    """
    Implements the tab content for data file handling
    """

    statusMessage = QtCore.pyqtSignal(['QString'])
    proceedAnalysis = QtCore.pyqtSignal()

    def __init__(self, parent=None, input_files=None, dut_names=None):
        super(SetupTab, self).__init__(parent)

        # Make variable for data from DataTab
        self.data = None

        # Make tuple of specs of each dut
        self._dut_specs = ('z_pos', 'rotation', 'pitch_col', 'pitch_row', 'n_cols', 'n_rows', 'thickness')

        # Make dict for specs of each dut
        self.dut_data = {}

        # Make dict of all data input widgets
        self._dut_widgets = {}
        self._type_widgets = {}
        self._create_widgets = {}

        # Make dict for dut types
        self.dut_types = {}

        # Load predefined dut types
        try:
            self.dut_types = json.load(open('dut_types.txt'))
        except IOError:
            pass

        self._setup()

    def input_data(self, data):

        self.data = data

        self._init_tabs()
        self._handle_dut_types()

    def _setup(self):
        # Draw area
        left_widget = QtWidgets.QWidget()
        self.draw = QtWidgets.QHBoxLayout()
        l = QtWidgets.QLabel('Wow. Great generic plotting of telescope!')
        self.draw.addWidget(l)
        left_widget.setLayout(self.draw)

        # Area for tabs for dut setup
        layout_right = QtWidgets.QVBoxLayout()
        layout_tabs = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(375)
        layout_tabs.addWidget(self.tabs)
        layout_right.addLayout(layout_tabs)
        # Proceed button
        button_ok = QtWidgets.QPushButton('OK')
        button_ok.clicked.connect(lambda: self.get_dut_specs())
        layout_right.addWidget(button_ok)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(layout_right)
        # Split plot and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 5)
        widget_splitter.setChildrenCollapsible(False)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def _init_tabs(self):

        self.tabs.clear()
        for i, dut in enumerate(self.data['dut_names']):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()
            layout.setSpacing(30)
            layout.addSpacing(15)
            widget.setLayout(layout)

            # Add sub layouts for spacing
            l_1 = QtWidgets.QVBoxLayout()
            l_1.setSpacing(15)
            l_2 = QtWidgets.QVBoxLayout()
            l_2.setSpacing(15)

            # Add layouts for setup and dut specs
            l_z = QtWidgets.QHBoxLayout()
            l_rot = QtWidgets.QHBoxLayout()
            l_types = QtWidgets.QHBoxLayout()
            l_pitch = QtWidgets.QHBoxLayout()
            l_pitch_sub = QtWidgets.QVBoxLayout()
            l_pixels = QtWidgets.QHBoxLayout()
            l_pixels_sub = QtWidgets.QVBoxLayout()
            l_thickness = QtWidgets.QHBoxLayout()
            l_create = QtWidgets.QHBoxLayout()
            l_z.addSpacing(10)
            l_rot.addSpacing(10)
            l_types.addSpacing(10)
            l_pitch.addSpacing(10)
            l_pixels.addSpacing(10)
            l_thickness.addSpacing(10)
            l_create.addSpacing(10)

            # Make dut type selection layout
            label_type = QtWidgets.QLabel('Select DUT type')
            cb_type = QtWidgets.QComboBox()
            cb_type.addItems(self.dut_types.keys())
            button_type = QtWidgets.QPushButton('Set')
            button_type.clicked.connect(
                lambda: self._set_dut_type(self.data['dut_names'][self.tabs.currentIndex()],
                                           self._type_widgets[self.data['dut_names'][self.tabs.currentIndex()]]['combo_t'].currentText()))
            checkbox_type = QtWidgets.QCheckBox()
            checkbox_type.toggled.connect(lambda:
                                          self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))
            label_type.setDisabled(True)
            cb_type.setDisabled(True)
            button_type.setDisabled(True)
            checkbox_type.setDisabled(True)

            # Make dut type creation layout
            label_create = QtWidgets.QLabel('Create DUT type')
            edit_create = QtWidgets.QLineEdit()
            edit_create.setPlaceholderText('Enter name')
            button_create = QtWidgets.QPushButton('Create')
            checkbox_create = QtWidgets.QCheckBox()
            checkbox_create.setToolTip('Create custom DUT type')
            checkbox_create.toggled.connect(lambda:
                                            self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))
            label_create.setDisabled(True)
            button_create.setDisabled(True)
            edit_create.setDisabled(True)

            # Make widgets for input parameters
            label_z = QtWidgets.QLabel('z-Position / ' + u'\u03BC' + 'm :')
            edit_z = QtWidgets.QLineEdit()
            label_rot = QtWidgets.QLabel('Rotation / ' + u'\u00B0' + ' :')
            edit_rot = QtWidgets.QLineEdit()
            label_pitch = QtWidgets.QLabel('Pixel pitch / ' + u'\u03BC' + 'm :')
            edit_pitch_col = QtWidgets.QLineEdit()
            edit_pitch_col.setPlaceholderText('Column')
            edit_pitch_row = QtWidgets.QLineEdit()
            edit_pitch_row.setPlaceholderText('Row')
            label_pixels = QtWidgets.QLabel('Number of pixels :')
            edit_pixels_col = QtWidgets.QLineEdit()
            edit_pixels_col.setPlaceholderText('Column')
            edit_pixels_row = QtWidgets.QLineEdit()
            edit_pixels_row.setPlaceholderText('Row')
            label_thickness = QtWidgets.QLabel('Thickness / ' + u'\u03BC' + 'm :')
            edit_thickness = QtWidgets.QLineEdit()

            # Add widgets to layouts
            l_z.addWidget(label_z)
            l_z.addStretch(1)
            l_z.addWidget(edit_z)
            l_rot.addWidget(label_rot)
            l_rot.addStretch(1)
            l_rot.addWidget(edit_rot)
            l_types.addWidget(checkbox_type)
            l_types.addWidget(label_type)
            l_types.addStretch(1)
            l_types.addWidget(cb_type)
            l_types.addWidget(button_type)
            l_pitch.addWidget(label_pitch)
            l_pitch.addStretch(1)
            l_pitch_sub.addWidget(edit_pitch_col)
            l_pitch_sub.addWidget(edit_pitch_row)
            l_pitch.addLayout(l_pitch_sub)
            l_pitch.setAlignment(label_pitch, QtCore.Qt.AlignTop)
            l_pitch.setAlignment(l_pitch_sub, QtCore.Qt.AlignTop)
            l_pixels.addWidget(label_pixels)
            l_pixels.addStretch(1)
            l_pixels_sub.addWidget(edit_pixels_col)
            l_pixels_sub.addWidget(edit_pixels_row)
            l_pixels.addLayout(l_pixels_sub)
            l_pixels.setAlignment(label_pixels, QtCore.Qt.AlignTop)
            l_pixels.setAlignment(l_pixels_sub, QtCore.Qt.AlignTop)
            l_thickness.addWidget(label_thickness)
            l_thickness.addStretch(1)
            l_thickness.addWidget(edit_thickness)
            l_create.addWidget(checkbox_create)
            l_create.addWidget(label_create)
            l_create.addStretch(1)
            l_create.addWidget(edit_create)
            l_create.addWidget(button_create)

            # Add to sub layouts
            l_1.addLayout(l_z)
            l_1.addLayout(l_rot)
            l_2.addLayout(l_types)
            l_2.addLayout(l_pitch)
            l_2.addLayout(l_pixels)
            l_2.addLayout(l_thickness)
            l_2.addLayout(l_create)

            # Add to main layout
            layout.addWidget(QtWidgets.QLabel('Setup'))
            layout.addLayout(l_1)
            layout.addWidget(QtWidgets.QLabel('Properties'))
            layout.addLayout(l_2)
            layout.addStretch(0)

            # z_pos of 1. DUT is origin: 0
            if i == 0:
                edit_z.setText('0')
                # edit_z.setReadOnly(True)
                edit_z.setDisabled(True)
                edit_z.setToolTip('First DUT defines origin of coordinate system')

            edit_widgets = (edit_z, edit_rot, edit_pitch_col, edit_pitch_row,
                            edit_pixels_col, edit_pixels_row, edit_thickness)

            self._dut_widgets[dut] = {}
            for j, spec in enumerate(self._dut_specs):
                self._dut_widgets[dut][spec] = edit_widgets[j]

            self._type_widgets[dut] = {'label_t': label_type, 'check_t': checkbox_type,
                                       'combo_t': cb_type, 'button_t': button_type}
            self._create_widgets[dut] = {'label_c': label_create, 'check_c': checkbox_create,
                                         'edit_c': edit_create, 'button_c': button_create}

            self.tabs.addTab(widget, dut)

    def get_dut_specs(self):

        self.dut_data = {}
        for dut in self.data['dut_names']:

            self.dut_data[dut] = {}
            for spec in self._dut_specs:
                try:
                    if spec == 'n_cols' or spec == 'n_rows':
                        t = int
                        self.dut_data[dut][spec] = int(self._dut_widgets[dut][spec].text())
                    else:
                        t = float
                        self.dut_data[dut][spec] = float(self._dut_widgets[dut][spec].text())
                except ValueError:
                    logging.error('%s of %s must be %s' % (spec, dut, str(t)))
                    pass

    def _handle_dut_types(self, dut=None, mode='r'):

        if len(self.dut_types) > 0:
            for dut_name in self.data['dut_names']:
                self._type_widgets[dut_name]['check_t'].setDisabled(False)

        if dut is not None:
            if self._type_widgets[dut]['check_t'].isChecked():
                for key in self._type_widgets[dut].keys():
                    self._type_widgets[dut][key].setDisabled(False)
            else:
                for key in self._type_widgets[dut].keys():
                    if key is not 'check_t':
                        self._type_widgets[dut][key].setDisabled(True)
                self._set_dut_type(self.data['dut_names'][self.tabs.currentIndex()], None)

            if self._create_widgets[dut]['check_c'].isChecked():
                for key in self._create_widgets[dut].keys():
                    self._create_widgets[dut][key].setDisabled(False)
            else:
                for key in self._create_widgets[dut].keys():
                    if key is not 'check_c':
                        self._create_widgets[dut][key].setDisabled(True)

    def _set_dut_type(self, dut, dut_type):

        for spec in self._dut_specs:
            if spec is 'z_pos' or spec is 'rotation':
                continue
            else:
                if dut_type is not None:
                    self._dut_widgets[dut][spec].setText(str(self.dut_types[dut_type][spec]))
                else:
                    self._dut_widgets[dut][spec].clear()

    def _emit_message(self, message):
        """
        Emits statusMessage signal with message
        """

        self.statusMessage.emit(message)
