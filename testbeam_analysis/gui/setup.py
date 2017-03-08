import logging
import json
from PyQt5 import QtCore, QtWidgets, QtGui


class SetupTab(QtWidgets.QWidget):
    """
    Implements the tab content for handling the setup of the detectors and their properties
    """

    statusMessage = QtCore.pyqtSignal(['QString'])
    proceedAnalysis = QtCore.pyqtSignal()

    def __init__(self, parent=None, input_files=None, dut_names=None):
        super(SetupTab, self).__init__(parent)

        # Make variable for data from DataTab
        self.data = None

        # Make tuple of properties of each dut
        self._dut_props = ('z_pos', 'rotation', 'pitch_col', 'pitch_row', 'n_cols', 'n_rows', 'thickness')

        # Make dict for properties of each dut
        self.dut_data = {}

        # Make several dicts which contain widgets to access them within the SetupTab
        # Make dict of all data input widgets
        self._dut_widgets = {}

        # Make dict for all dut type setting/handling widgets
        self._type_widgets = {}

        # Make dict for all create dut type widgets
        self._create_widgets = {}

        # Make dict for buttons for deleting props or setting globally
        self._buttons = {}

        # Make dict for dut types
        self._dut_types = {}

        # Load predefined dut types from file
        try:
            self._dut_types = json.load(open('dut_types.txt'))
        except IOError:
            pass

        # Setup the UI
        self._setup()

    def _setup(self):
        # Draw area for generic plotting of setup
        left_widget = QtWidgets.QWidget()
        self.draw = QtWidgets.QHBoxLayout()
        l = QtWidgets.QLabel('Wow. Magnificent generic plotting of telescope!')
        self.draw.addWidget(l)
        left_widget.setLayout(self.draw)

        # Area for setup tabs for each dut
        layout_right = QtWidgets.QVBoxLayout()
        layout_tabs = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(400)
        layout_tabs.addWidget(self.tabs)
        layout_right.addLayout(layout_tabs)
        layout_right.addSpacing(10)

        # Proceed button
        self.button_ok = QtWidgets.QPushButton('OK')
        self.button_ok.clicked.connect(lambda: self._get_properties())
        self.button_ok.setDisabled(True)
        layout_right.addWidget(self.button_ok)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(layout_right)

        # Split plot and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 6)
        widget_splitter.setChildrenCollapsible(False)

        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def input_data(self, data):
        """
        Method to get the input data from the DataTab and initiate tabs for each dut.

        :param data: dict; containing output_folder and lists of dut_names and paths for each dut
        """
        # Set data
        self.data = data

        # Initiate tabs and initially check and handle each tabs content
        self._init_tabs()
        self._handle_dut_types()
        self._check_input()

    def _init_tabs(self):
        """
        Initializes the setup tabs and their widget contents for each dut in the dut_names list from DataTab
        """

        # Clear the QTabwidget which holds each duts tab to allow updating the data from DataTab
        self.tabs.clear()

        for i, dut in enumerate(self.data['dut_names']):

            # Main widget for each tab
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()
            layout.setSpacing(30)
            layout.addSpacing(10)
            widget.setLayout(layout)

            # Add sub layouts for spacing
            l_1 = QtWidgets.QVBoxLayout()
            l_1.setSpacing(20)
            l_2 = QtWidgets.QVBoxLayout()
            l_2.setSpacing(20)

            # Add layouts for setup and dut properties
            l_z = QtWidgets.QHBoxLayout()
            l_rot = QtWidgets.QHBoxLayout()
            l_types = QtWidgets.QHBoxLayout()
            l_pitch = QtWidgets.QHBoxLayout()
            l_pitch_sub = QtWidgets.QVBoxLayout()
            l_pixels = QtWidgets.QHBoxLayout()
            l_pixels_sub = QtWidgets.QVBoxLayout()
            l_thickness = QtWidgets.QHBoxLayout()
            l_create = QtWidgets.QHBoxLayout()
            l_buttons = QtWidgets.QHBoxLayout()
            l_z.addSpacing(10)
            l_rot.addSpacing(10)
            l_types.addSpacing(10)
            l_pitch.addSpacing(10)
            l_pixels.addSpacing(10)
            l_thickness.addSpacing(10)
            l_create.addSpacing(10)

            # Make dut type selection layout
            label_type = QtWidgets.QLabel('Set DUT type')
            cb_type = QtWidgets.QComboBox()
            cb_type.addItems(self._dut_types.keys())
            button_type = QtWidgets.QPushButton('Set')
            checkbox_type = QtWidgets.QCheckBox()
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
            checkbox_create.setToolTip('Permanently create custom DUT type')
            label_create.setDisabled(True)
            button_create.setDisabled(True)
            edit_create.setDisabled(True)

            # Make widgets for input parameters
            label_z = QtWidgets.QLabel('z-Position / ' + u'\u03BC' + 'm :')
            edit_z = QtWidgets.QLineEdit()

            # z_pos of 1. DUT is origin and set to 0
            if i == 0:
                edit_z.setText('0')
                # edit_z.setReadOnly(True)
                edit_z.setDisabled(True)
                edit_z.setToolTip('First DUT defines origin of coordinate system')

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

            # Make buttons at the bottom to clear input and apply
            button_clear = QtWidgets.QPushButton('Clear properties')
            button_clear.setToolTip('Clear current set of properties')
            button_glob = QtWidgets.QPushButton('Globalize properties')
            button_glob.setToolTip('Set current set of input parameters for all DUTs')

            # Disable globalise button in case of only 1 dut
            if len(self.data['dut_names']) == 1:
                button_glob.setDisabled(True)
                button_glob.setToolTip('')

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
            l_buttons.addWidget(button_glob)
            l_buttons.addWidget(button_clear)

            # Add layouts to sub layouts
            l_1.addLayout(l_z)
            l_1.addLayout(l_rot)
            l_2.addLayout(l_types)
            l_2.addLayout(l_pitch)
            l_2.addLayout(l_pixels)
            l_2.addLayout(l_thickness)
            l_2.addLayout(l_create)
            l_2.addStretch(0)
            l_2.addLayout(l_buttons)

            # Add sub layouts to main layout
            layout.addWidget(QtWidgets.QLabel('Setup'))
            layout.addLayout(l_1)
            layout.addWidget(QtWidgets.QLabel('Properties'))
            layout.addLayout(l_2)
            layout.addSpacing(10)

            # Make tuple of all QLineEdit widgets
            edit_widgets = (edit_z, edit_rot, edit_pitch_col, edit_pitch_row,
                            edit_pixels_col, edit_pixels_row, edit_thickness)

            # Add these QLineEdit widgets to dict with respective dut as key to access the user input
            self._dut_widgets[dut] = {}
            for j, prop in enumerate(self._dut_props):
                self._dut_widgets[dut][prop] = edit_widgets[j]

            # Add all dut type setting/handling related widgets to a dict with respective dut as key
            self._type_widgets[dut] = {'label_t': label_type, 'check_t': checkbox_type,
                                       'combo_t': cb_type, 'button_t': button_type}

            # Add all dut type creating related widgets to a dict with respective dut as key
            self._create_widgets[dut] = {'label_c': label_create, 'check_c': checkbox_create,
                                         'edit_c': edit_create, 'button_c': button_create}

            # Add all button widgets to a dict with respective dut as key
            self._buttons[dut] = {'global': button_glob, 'clear': button_clear}

            # Add tab to QTabWidget
            self.tabs.addTab(widget, dut)

        # Connect tabs
        self.tabs.currentChanged.connect(lambda tab: self._handle_dut_types(self.data['dut_names'][tab]))

        # Connect widgets of all tabs
        for dut in self.data['dut_names']:

            # Connect set type button
            self._type_widgets[dut]['button_t'].clicked.connect(
                lambda: self._set_properties(
                    self.data['dut_names'][self.tabs.currentIndex()],
                    self._type_widgets[self.data['dut_names'][self.tabs.currentIndex()]]['combo_t'].currentText()))

            # Connect set type checkbox
            self._type_widgets[dut]['check_t'].toggled.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect create type checkbox
            self._create_widgets[dut]['check_c'].toggled.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect create type edit
            self._create_widgets[dut]['edit_c'].textChanged.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect create type button
            self._create_widgets[dut]['button_c'].clicked.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()], 'c'))

            # Connect clear buttons
            self._buttons[dut]['clear'].clicked.connect(
                lambda: self._set_properties(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect globalize buttons
            self._buttons[dut]['global'].clicked.connect(
                lambda: self._set_properties(self.data['dut_names'][self.tabs.currentIndex()], 'global'))

            # Connect all input QLineEdit widgets
            for prop in self._dut_props:
                for x in [lambda: self._check_input(),
                          lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()])]:
                    self._dut_widgets[dut][prop].textChanged.connect(x)

    def _get_properties(self, custom=None):
        """
        Method to read input parameters of each dut from the UI. If custom is not None, only the specific
        dut properties of the custom dut type "custom" are read. The dut type "custom" is added to self._dut_types
        and permanently saved to the dut_types file.

        :param custom: str; name of new custom dut type
        """
        # Clear dut data dict before reading
        self.dut_data = {}

        # Read input parameters of all duts
        if custom is None:

            for dut in self.data['dut_names']:

                self.dut_data[dut] = {}
                for prop in self._dut_props:

                    if prop in ['n_cols', 'n_rows']:
                        self.dut_data[dut][prop] = int(self._dut_widgets[dut][prop].text())
                    else:
                        self.dut_data[dut][prop] = float(self._dut_widgets[dut][prop].text())

        # Read only dut properties of custom dut type
        else:

            self._dut_types[custom] = {}
            current_dut = self.data['dut_names'][self.tabs.currentIndex()]

            for prop in self._dut_props:

                # Exclude reading of non-specific dut properties
                if prop not in ['z_pos', 'rotation']:
                    if prop in ['n_cols', 'n_rows']:
                        self._dut_types[custom][prop] = int(self._dut_widgets[current_dut][prop].text())
                    else:
                        self._dut_types[custom][prop] = float(self._dut_widgets[current_dut][prop].text())

            # Clear QLineEdit of tab where the custom dut was created
            self._create_widgets[current_dut]['edit_c'].clear()

            # Safe updated self._dut_types dict to file and reload
            try:
                json.dump(self._dut_types, open('dut_types.txt', 'wb'))
                self._dut_types = json.load(open('dut_types.txt'))
                message = 'Successfully created DUT type "%s" and added to predefined DUT types' % custom
                self._emit_message(message)

                # Update selection of dut types in each dut tabs combobox
                for dut in self.data['dut_names']:
                    self._type_widgets[dut]['combo_t'].clear()
                    self._type_widgets[dut]['combo_t'].addItems(self._dut_types.keys())

            except IOError:
                message = 'Could not reload file with predefined DUT types'
                self._emit_message(message)
                pass

    def _handle_dut_types(self, dut=None, mode='h'):
        """
        Method to handle predefined dut type selection and creation of new types of all tabs.

        :param dut: None or str; name of dut from dut_names list; if None, check for dut types and enable selection
        :param mode: str; "h" or "c"; h for handle mode, c for create mode
        """

        # Check if dut types selectable and enable respective widgets
        if len(self._dut_types) > 0:
            for dut_name in self.data['dut_names']:
                self._type_widgets[dut_name]['check_t'].setDisabled(False)

        if dut is not None:
            # Go through widgets of respective dut tab and handle behavior
            if mode is 'h':
                if self._type_widgets[dut]['check_t'].isChecked():
                    for key in self._type_widgets[dut].keys():
                        self._type_widgets[dut][key].setDisabled(False)
                else:
                    for key in self._type_widgets[dut].keys():
                        if key is not 'check_t':
                            self._type_widgets[dut][key].setDisabled(True)

                if self._create_widgets[dut]['check_c'].isChecked():
                    new_type = self._create_widgets[dut]['edit_c'].text()
                    for key in self._create_widgets[dut].keys():
                        if key is 'button_c':
                            if len(new_type) > 0:
                                if new_type not in self._dut_types.keys():
                                    self._check_input(['z_pos', 'rotation'])
                                    self._create_widgets[dut][key].setText('Create')
                                else:
                                    self._create_widgets[dut][key].setText('Overwrite')
                                    message = 'Predefined DUT type "%s" will be overwritten!' % new_type
                                    self._emit_message(message)
                            else:
                                self._create_widgets[dut][key].setDisabled(True)
                        else:
                            self._create_widgets[dut][key].setDisabled(False)
                else:
                    for key in self._create_widgets[dut].keys():
                        if key is not 'check_c':
                            self._create_widgets[dut][key].setDisabled(True)
                            if key is 'edit_c':
                                self._create_widgets[dut][key].clear()
            if mode is 'c':
                new_dut = self._create_widgets[dut]['edit_c'].text()
                self._get_properties(new_dut)

    def _set_properties(self, dut, dut_type=None):
        """
        Method to set the properties of a dut by writing text to its respective self._dut_widgets.

        :param dut: The dut from self.data['dut_names'] whose properties are set/cleared
        :param dut_type: Can be either None or a predefined type from self._dut_types; if None, clear props

        """

        for prop in self._dut_props:

            # Loop over properties, only set dut specific props
            if prop not in ['z_pos', 'rotation']:

                if dut_type is not None:
                    # List of all dut types in duts combobox
                    dut_types = [self._type_widgets[dut]['combo_t'].itemText(i)
                                 for i in range(self._type_widgets[dut]['combo_t'].count())]

                    # If type is predefined
                    if dut_type in dut_types :
                        self._dut_widgets[dut][prop].setText(str(self._dut_types[dut_type][prop]))

                    # If global, current properties are set for all duts
                    elif dut_type is 'global':
                        l = []
                        for dut_name in self.data['dut_names']:
                            if dut_name is not dut:
                                l.append(dut_name)
                                self._dut_widgets[dut_name][prop].setText(self._dut_widgets[dut][prop].text())
                        if len(l) > 0:
                            message = 'Set properties of %s as global properties for %s' % (dut, str(', ').join(l))
                            self._emit_message(message)
                else:
                    self._dut_widgets[dut][prop].clear()

    def _check_input(self, skip_props=None):
        """
        Method that checks all input related widgets of the UI for the correct types of input parameters.
        In regard of the correct type, the behavior of respective buttons etc. is handled. If all input of all
        tabs is complete and correct, the button to proceed to next step is enabled, if not it is disabled.

        :param skip_props: list; properties that are skipped during the check to only check dut-specific input
        """

        # Store correctness status of all input and list of incorrect input parameters
        broken = False
        broken_input = []

        # Make list of dut properties
        properties = list(self._dut_props)

        # Skip checking properties in skip_props by removing them from list
        if skip_props is not None:
            for skip in skip_props:
                properties.remove(skip)

        # Check all props of all duts
        for dut in self.data['dut_names']:
            for prop in properties:

                # Get input text of respective property
                in_put = self._dut_widgets[dut][prop].text()

                # Input not empty
                if len(in_put) > 0:

                    # Check whether required conversions of types are possible
                    try:
                        if prop in ['n_cols', 'n_rows']:
                            _ = int(in_put)
                        else:
                            _ = float(in_put)

                    except ValueError:
                        broken = True
                        if dut not in broken_input:
                            broken_input.append(dut)
                        break

                # Input is empty
                else:
                    broken = True
                    if dut not in broken_input:
                        broken_input.append(dut)
                    break

            # Check entire data of dut
            if skip_props is None:

                # TabIcon related variable
                style = QtWidgets.qApp.style()

                # Set TabIcons to respective duts tab to visualize whether input parameters are correct
                if dut in broken_input:
                    icon = style.standardIcon(style.SP_DialogCancelButton)
                    self.tabs.setTabIcon(self.data['dut_names'].index(dut), icon)
                    self.tabs.setTabToolTip(self.data['dut_names'].index(dut), 'Fill in required information')

                else:
                    icon = style.standardIcon(style.SP_DialogApplyButton)
                    self.tabs.setTabIcon(self.data['dut_names'].index(dut), icon)
                    self.tabs.setTabToolTip(self.data['dut_names'].index(dut), 'Ready')

                self._create_widgets[dut]['button_c'].setDisabled(broken)

            else:
                if dut in broken_input:
                    self._create_widgets[dut]['button_c'].setDisabled(True)
                else:
                    self._create_widgets[dut]['button_c'].setDisabled(False)


        # Set the status of the proceed button
        if skip_props is None:
            self.button_ok.setDisabled(broken)

    def _emit_message(self, message):
        """
        Emits statusMessage signal with message
        """

        self.statusMessage.emit(message)
