import yaml
from PyQt5 import QtCore, QtWidgets, QtGui


class SetupTab(QtWidgets.QWidget):
    """
    Implements the tab content for handling the setup of the detectors and their properties
    """

    statusMessage = QtCore.pyqtSignal('QString')
    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent=None, input_files=None, dut_names=None):
        super(SetupTab, self).__init__(parent)

        # Make variable for data from DataTab and output data
        self.data = None

        # Tab widgets
        self.tw = {}

        # Make list of tabs that will be enabled after proceedAnalysis signal of this class
        self.tab_list = ['Noisy Pixel']

        # Make tuple of properties of each dut
        self._dut_props = ('z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma', 'pitch_col',
                           'pitch_row', 'n_cols', 'n_rows', 'thickness', 'rad_length')

        # Make dict for properties of each dut
        self.dut_data = {}

        # Make several dicts which contain widgets to access them within the SetupTab
        # Make dict of all data input widgets
        self._dut_widgets = {}

        # Make dict for all dut type setting/handling widgets
        self._type_widgets = {}

        # Make dict for all handle dut type widgets
        self._handle_widgets = {}

        # Make dict for buttons for deleting props or setting globally
        self._buttons = {}

        # Make dict for dut types
        self._dut_types = {}

        # Load predefined dut types from file
        try:
            with open('dut_types.yaml', 'r') as f_read:
                self._dut_types = yaml.load(f_read)
        except IOError:
            pass

        # Setup the UI
        self._setup()

    def _setup(self):
        # Draw area for generic plotting of setup
        self.left_widget = QtWidgets.QWidget()
        self.left_widget.setMinimumWidth(475)
        self.draw = QtWidgets.QVBoxLayout()
        self.left_widget.setLayout(self.draw)

        # Area for setup tabs for each dut
        layout_right = QtWidgets.QVBoxLayout()
        layout_tabs = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(475)
        layout_tabs.addWidget(self.tabs)
        layout_right.addLayout(layout_tabs)
        layout_right.addSpacing(10)

        # Proceed button
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(lambda: self._handle_input())
        self.btn_ok.setDisabled(True)
        layout_right.addWidget(self.btn_ok)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(layout_right)

        # Split plot and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(self.left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 10)
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
        self._init_setup(self.data['dut_names'])
        self._init_tabs()
        self._handle_dut_types()
        self._check_input()

    def _init_setup(self, dut_names):

        # Remove old SetupPainter if new data comes in
        for i in reversed(range(self.draw.count())):
            w = self.draw.itemAt(i).widget()
            if isinstance(w, SetupPainter):
                self.draw.removeWidget(w)
                w.setParent(None)

        label_side = QtWidgets.QLabel('Schematic side-view:')
        label_top = QtWidgets.QLabel('Schematic top-view:')

        self.side_view = SetupPainter(dut_names, self.left_widget)
        self.side_view.draw_coordinate_system(axes=['z', 'x', 'y'])
        self.top_view = SetupPainter(dut_names, self.left_widget)
        self.top_view.draw_coordinate_system(axes=['z', 'y', '-x'])

        self.draw.addWidget(label_top)
        self.draw.addWidget(self.top_view)
        self.draw.addWidget(label_side)
        self.draw.addWidget(self.side_view)

    def _init_tabs(self):
        """
        Initializes the setup tabs and their widget contents for each dut in the dut_names list from DataTab
        """

        # Clear the QTabwidget which holds each duts tab to allow updating the data from DataTab
        self.tabs.clear()

        # Spacing related numbers
        h_space = 10
        v_space = 25
        label_width = 175  # Fixed width for alignment

        for i, dut in enumerate(self.data['dut_names']):

            # Main widget for each tab
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()
            layout.setSpacing(v_space)
            layout.addSpacing(v_space)
            widget.setLayout(layout)

            # Add sub layouts for spacing
            sl_1 = QtWidgets.QHBoxLayout()
            sl_1.addSpacing(h_space)
            sl_2 = QtWidgets.QHBoxLayout()
            sl_2.addSpacing(h_space)

            # Add layout for setup and dut properties
            layout_setup = QtWidgets.QGridLayout()
            layout_props = QtWidgets.QGridLayout()
            layout_buttons = QtWidgets.QHBoxLayout()

            # Add to sub layouts
            sl_1.addLayout(layout_setup)
            sl_2.addLayout(layout_props)

            # Make dut type selection layout
            label_type = QtWidgets.QLabel('Set DUT type')
            label_type.setFixedWidth(label_width)
            cb_type = QtWidgets.QComboBox()
            cb_type.addItems(self._dut_types.keys())
            cb_type.setDisabled(True)
            button_type = QtWidgets.QPushButton('Set')
            button_type.setDisabled(True)

            # Make dut type creation layout
            checkbox_handle = QtWidgets.QCheckBox()
            checkbox_handle.setText('Create DUT type')
            checkbox_handle.setFixedWidth(label_width)
            checkbox_handle.setToolTip('Create or remove (:type) custom DUT type')
            edit_handle = QtWidgets.QLineEdit()
            edit_handle.setPlaceholderText('Enter name')
            edit_handle.setDisabled(True)
            button_handle = QtWidgets.QPushButton('Create')
            button_handle.setDisabled(True)

            # Make widgets for input parameters
            label_z = QtWidgets.QLabel('z-Position / ' + u'\u03BC' + 'm :')
            label_z.setFixedWidth(label_width)
            edit_z = QtWidgets.QLineEdit()

            # z_positions of 1. DUT is origin and set to 0
            if i == 0:
                edit_z.setText('0')
                # edit_z.setReadOnly(True)
                edit_z.setDisabled(True)
                edit_z.setToolTip('First DUT defines origin of coordinate system')

            label_rot = QtWidgets.QLabel('Rotation / mRad:')
            label_rot.setFixedWidth(label_width)
            edit_alpha = QtWidgets.QLineEdit()
            edit_alpha.setToolTip('Rotation around x-axis')
            edit_alpha.setPlaceholderText(u'\u03B1' + ' = 0')
            edit_beta = QtWidgets.QLineEdit()
            edit_beta.setToolTip('Rotation around y-axis')
            edit_beta.setPlaceholderText(u'\u03B2' + ' = 0')
            edit_gamma = QtWidgets.QLineEdit()
            edit_gamma.setToolTip('Rotation around z-axis. Not shown in setup')
            edit_gamma.setPlaceholderText(u'\u03B3' + ' = 0')
            label_pitch = QtWidgets.QLabel('Pixel pitch / ' + u'\u03BC' + 'm :')
            label_pitch.setFixedWidth(label_width)
            edit_pitch_col = QtWidgets.QLineEdit()
            edit_pitch_col.setPlaceholderText('Column')
            edit_pitch_row = QtWidgets.QLineEdit()
            edit_pitch_row.setPlaceholderText('Row')
            label_pixels = QtWidgets.QLabel('Number of pixels :')
            label_pixels.setFixedWidth(label_width)
            edit_pixels_col = QtWidgets.QLineEdit()
            edit_pixels_col.setPlaceholderText('Column')
            edit_pixels_row = QtWidgets.QLineEdit()
            edit_pixels_row.setPlaceholderText('Row')
            label_thickness = QtWidgets.QLabel('Thickness / ' + u'\u03BC' + 'm :')
            label_thickness.setToolTip('Thickness of sensor or compound (e.g. sensor + PCB + ...)')
            label_thickness.setFixedWidth(label_width)
            edit_thickness = QtWidgets.QLineEdit()
            label_rad = QtWidgets.QLabel('Radiation length / ' + u'\u03BC' + 'm :')
            label_rad.setToolTip('Radiation length of sensor or compound')
            label_rad.setFixedWidth(label_width)
            edit_rad = QtWidgets.QLineEdit()
            edit_rad.setPlaceholderText(u'X\u2080' + ' = 0 ' + u'\u03BC' + 'm')

            # Make buttons at the bottom to clear input and apply
            button_clear = QtWidgets.QPushButton('Clear properties')
            button_clear.setToolTip('Clear current set of properties')
            button_glob = QtWidgets.QPushButton('Set properties for all DUTs')
            button_glob.setToolTip('Set current set of properties for all DUTs')

            # Disable globalise button in case of only 1 dut
            if len(self.data['dut_names']) == 1:
                button_glob.setDisabled(True)
                button_glob.setToolTip('')

            # Add widgets to layouts
            # Setup layout
            layout_setup.addWidget(label_z, 0, 0, 1, 1)
            layout_setup.addItem(QtWidgets.QSpacerItem(h_space, v_space), 0, 1, 1, 1)
            layout_setup.addWidget(edit_z, 0, 2, 1, 3)
            layout_setup.addWidget(label_rot, 1, 0, 1, 1)
            layout_setup.addItem(QtWidgets.QSpacerItem(h_space, v_space), 1, 1, 1, 1)
            layout_setup.addWidget(edit_alpha, 1, 2, 1, 1)
            layout_setup.addWidget(edit_beta, 1, 3, 1, 1)
            layout_setup.addWidget(edit_gamma, 1, 4, 1, 1)
            # Properties layout
            layout_props.addWidget(label_type, 0, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 0, 1, 1, 1)
            layout_props.addWidget(cb_type, 0, 2, 1, 1)
            layout_props.addWidget(button_type, 0, 3, 1, 1)
            layout_props.addWidget(label_pitch, 1, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 1, 1, 1, 1)
            layout_props.addWidget(edit_pitch_col, 1, 2, 1, 1)
            layout_props.addWidget(edit_pitch_row, 1, 3, 1, 1)
            layout_props.addWidget(label_pixels, 2, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 2, 1, 1, 1)
            layout_props.addWidget(edit_pixels_col, 2, 2, 1, 1)
            layout_props.addWidget(edit_pixels_row, 2, 3, 1, 1)
            layout_props.addWidget(label_thickness, 3, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 3, 1, 1, 1)
            layout_props.addWidget(edit_thickness, 3, 2, 1, 2)
            layout_props.addWidget(label_rad, 4, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 4, 1, 1, 1)
            layout_props.addWidget(edit_rad, 4, 2, 1, 2)
            layout_props.addWidget(checkbox_handle, 5, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 5, 1, 1, 1)
            layout_props.addWidget(edit_handle, 5, 2, 1, 1)
            layout_props.addWidget(button_handle, 5, 3, 1, 1)
            # Button layout
            layout_buttons.addWidget(button_glob)
            layout_buttons.addWidget(button_clear)

            # Add sub layouts to main layout
            layout.addWidget(QtWidgets.QLabel('Setup'))
            layout.addLayout(sl_1)
            layout.addWidget(QtWidgets.QLabel('Properties'))
            layout.addLayout(sl_2)
            layout.addLayout(layout_buttons)
            layout.addSpacing(h_space)

            # Make tuple of all QLineEdit widgets
            edit_widgets = (edit_z, edit_alpha, edit_beta, edit_gamma, edit_pitch_col,
                            edit_pitch_row, edit_pixels_col, edit_pixels_row, edit_thickness, edit_rad)

            # Add these QLineEdit widgets to dict with respective dut as key to access the user input
            self._dut_widgets[dut] = {}
            for j, prop in enumerate(self._dut_props):

                # Set QValidator to restrict input
                if prop in ['n_cols', 'n_rows']:
                    edit_widgets[j].setValidator(QtGui.QIntValidator()) #FIXME: Non-int input still possible e.g. 74.1
                else:
                    edit_widgets[j].setValidator(QtGui.QDoubleValidator())

                self._dut_widgets[dut][prop] = edit_widgets[j]

            # Add all dut type setting/handling related widgets to a dict with respective dut as key
            self._type_widgets[dut] = {'combo_t': cb_type, 'button_t': button_type}

            # Add all dut type creating related widgets to a dict with respective dut as key
            self._handle_widgets[dut] = {'check_h': checkbox_handle, 'edit_h': edit_handle, 'button_h': button_handle}

            # Add all button widgets to a dict with respective dut as key
            self._buttons[dut] = {'global': button_glob, 'clear': button_clear}

            # Add to tab widget dict
            self.tw[dut] = widget

            # Add tab to QTabWidget
            self.tabs.addTab(self.tw[dut], dut)

        # Connect tabs
        # self.tabs.currentChanged.connect(lambda tab: self._handle_dut_types(self.data['dut_names'][tab])) # FIXME

        # Connect widgets of all tabs
        for dut in self.data['dut_names']:

            # Connect set type button
            self._type_widgets[dut]['button_t'].clicked.connect(
                lambda: self._set_properties(
                    self.data['dut_names'][self.tabs.currentIndex()],
                    self._type_widgets[self.data['dut_names'][self.tabs.currentIndex()]]['combo_t'].currentText()))

            # Connect handle type checkbox
            self._handle_widgets[dut]['check_h'].toggled.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect handle type edit
            self._handle_widgets[dut]['edit_h'].textChanged.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect handle type button
            self._handle_widgets[dut]['button_h'].clicked.connect(
                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()], 'c'))

            # Connect clear buttons
            self._buttons[dut]['clear'].clicked.connect(
                lambda: self._set_properties(self.data['dut_names'][self.tabs.currentIndex()]))

            # Connect globalize buttons
            self._buttons[dut]['global'].clicked.connect(
                lambda: self._set_properties(self.data['dut_names'][self.tabs.currentIndex()], 'global'))

            # Connect all input QLineEdit widgets
            for prop in self._dut_props:

                if prop == 'rot_alpha':
                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda text: self.top_view.set_rotation(self.data['dut_names'][self.tabs.currentIndex()],
                                                                text))
                if prop == 'rot_beta':
                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda text: self.side_view.set_rotation(self.data['dut_names'][self.tabs.currentIndex()],
                                                                 text))
                if prop == 'z_positions':
                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda text: self.side_view.set_z_pos(self.data['dut_names'][self.tabs.currentIndex()],
                                                              text))
                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda: self.side_view.set_rotation(self.data['dut_names'][self.tabs.currentIndex()]))

                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda text: self.top_view.set_z_pos(self.data['dut_names'][self.tabs.currentIndex()],
                                                             text))
                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda: self.top_view.set_rotation(self.data['dut_names'][self.tabs.currentIndex()]))

                for x in [lambda: self._check_input(),
                          lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()])]:
                    self._dut_widgets[dut][prop].textChanged.connect(x)

    def _handle_input(self, custom=None):
        """
        Method to read input parameters of each dut from the UI. If custom is not None, only the specific
        dut properties of the custom dut type "custom" are read. The dut type "custom" is added to self._dut_types
        and permanently saved to the dut_types file.

        :param custom: str; name of new custom dut type
        """

        # Read input parameters and make list of of each parameter for all duts, then save to output data
        if custom is None:

            # Clear dut data dict before reading
            self.dut_data = {'rotations': [], 'n_pixels': [], 'pixel_size': []}

            # make tmp dict for pixel_size and n_pixel to add tuple of (n_cols, n_rows) etc. to output data
            tmp_rotation = {}
            tmp_n_pixel = {}
            tmp_pixel_size = {}

            for dut in self.data['dut_names']:
                for prop in self._dut_props:

                    # Initialize list for each property to append for each dut
                    if prop not in self.dut_data.keys():
                        self.dut_data[prop] = []

                    if prop in ['n_cols', 'n_rows']:
                        tmp_n_pixel[prop] = int(self._dut_widgets[dut][prop].text())
                    elif prop in ['pitch_col', 'pitch_row']:
                        tmp_pixel_size[prop] = float(self._dut_widgets[dut][prop].text())
                    elif prop in ['rot_alpha', 'rot_beta', 'rot_gamma', 'rad_length']:
                        # Default value for rotations and X_0 is 0
                        if not self._dut_widgets[dut][prop].text():
                            if prop == 'rad_length':
                                self.dut_data[prop].append(0.0)
                            else:
                                tmp_rotation[prop] = 0.0
                        else:
                            if prop == 'rad_length':
                                self.dut_data[prop].append(float(self._dut_widgets[dut][prop].text()))
                            else:
                                tmp_rotation[prop] = float(self._dut_widgets[dut][prop].text())
                    else:
                        self.dut_data[prop].append(float(self._dut_widgets[dut][prop].text()))

                self.dut_data['rotations'].append((tmp_rotation['rot_alpha'], tmp_rotation['rot_beta'],
                                                  tmp_rotation['rot_gamma']))
                self.dut_data['n_pixels'].append((tmp_n_pixel['n_cols'], tmp_n_pixel['n_rows']))
                self.dut_data['pixel_size'].append((tmp_pixel_size['pitch_col'], tmp_pixel_size['pitch_row']))

            # Add property lists to output data dict
            for key in self.dut_data.keys():
                if key not in ['n_cols', 'n_rows', 'pitch_col', 'pitch_row', 'rot_alpha', 'rot_beta', 'rot_gamma']:
                    self.data[key] = self.dut_data[key]

            self.proceedAnalysis.emit(self.tab_list)
            self._disable_tabs()

        # Read only dut properties of custom dut type or remove/overwrite predefined type
        else:
            current_dut = self.data['dut_names'][self.tabs.currentIndex()]
            remove = False

            if custom[0] == ':':
                custom = custom.split(':')[1]
                self._dut_types.pop(custom)
                remove = True

            else:
                self._dut_types[custom] = {}
                for prop in self._dut_props:

                    # Exclude reading of non-specific dut properties
                    if prop not in ['z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma']:
                        if prop in ['n_cols', 'n_rows']:
                            self._dut_types[custom][prop] = int(self._dut_widgets[current_dut][prop].text())
                        else:
                            try:
                                self._dut_types[custom][prop] = float(self._dut_widgets[current_dut][prop].text())
                            except ValueError:  # rad_length has default value 0
                                self._dut_types[custom][prop] = 0.0

            # Clear QLineEdit of tab where the custom dut was created or removed
            self._handle_widgets[current_dut]['edit_h'].clear()

            if remove:
                self._handle_widgets[current_dut]['button_h'].setText('Create')
                self._handle_widgets[current_dut]['check_h'].setText('Create DUT type')

            # Safe updated self._dut_types dict to file and reload
            try:
                with open('dut_types.yaml', 'w') as f_write:
                    yaml.dump(self._dut_types, f_write, default_flow_style=False)
                with open('dut_types.yaml', 'r') as f_read:
                    self._dut_types = yaml.load(f_read)
                if remove:
                    message = 'Successfully removed DUT type "%s" from predefined DUT types' % custom
                else:
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
        if dut is None:
            if self._dut_types:
                for dut_name in self.data['dut_names']:
                    for key in self._type_widgets[dut_name].keys():
                        self._type_widgets[dut_name][key].setDisabled(False)
            else:
                for dut_name in self.data['dut_names']:
                    for key in self._type_widgets[dut_name].keys():
                        self._type_widgets[dut_name][key].setToolTip('No predefined DUT types found!')

        elif dut is not None:

            # Go through widgets of respective dut tab and handle behavior
            if mode is 'h':

                if self._handle_widgets[dut]['check_h'].isChecked():

                    new_type = self._handle_widgets[dut]['edit_h'].text()

                    for key in self._handle_widgets[dut].keys():

                        if key is 'button_h':

                            self._handle_widgets[dut][key].setText('Create')
                            self._handle_widgets[dut]['check_h'].setText('Create DUT type')

                            if len(new_type) > 0:

                                if new_type[0] == ':':
                                    self._handle_widgets[dut][key].setText('Remove')
                                    self._handle_widgets[dut]['check_h'].setText('Remove DUT type')
                                    if new_type.split(':')[1] in self._dut_types.keys():
                                        self._handle_widgets[dut][key].setDisabled(False)
                                    else:
                                        self._handle_widgets[dut][key].setDisabled(True)

                                elif new_type not in self._dut_types.keys():
                                    self._check_input(['z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma'])

                                else:
                                    self._check_input(['z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma'])
                                    self._handle_widgets[dut][key].setText('Overwrite')
                                    self._handle_widgets[dut]['check_h'].setText('Overwrite DUT type')
                                    message = 'Predefined DUT type "%s" will be overwritten!' % new_type
                                    self._emit_message(message)
                            else:
                                self._handle_widgets[dut][key].setDisabled(True)
                                self._handle_widgets[dut]['check_h'].setText('Create DUT type')
                        else:
                            self._handle_widgets[dut][key].setDisabled(False)
                else:
                    self._handle_widgets[dut]['button_h'].setText('Create')
                    self._handle_widgets[dut]['check_h'].setText('Create DUT type')
                    for key in self._handle_widgets[dut].keys():
                        if key is not 'check_h':
                            self._handle_widgets[dut][key].setDisabled(True)
                            if key is 'edit_h':
                                self._handle_widgets[dut][key].clear()
            if mode is 'c':
                new_dut = self._handle_widgets[dut]['edit_h'].text()
                self._handle_input(new_dut)

    def _set_properties(self, dut, dut_type=None):
        """
        Method to set the properties of a dut by writing text to its respective self._dut_widgets.

        :param dut: The dut from self.data['dut_names'] whose properties are set/cleared
        :param dut_type: Can be either None or a predefined type from self._dut_types; if None, clear props

        """

        for prop in self._dut_props:

            # Loop over properties, only set dut specific props
            if prop not in ['z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma']:

                if dut_type is not None:
                    # List of all dut types in duts combobox
                    dut_types = [self._type_widgets[dut]['combo_t'].itemText(i)
                                 for i in range(self._type_widgets[dut]['combo_t'].count())]

                    # If type is predefined
                    if dut_type in dut_types:
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

                # FIXME: test-wise excluding rotations and X_0 from checking
                if prop in ['rot_alpha', 'rot_beta', 'rot_gamma', 'rad_length']:
                    continue

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

                if broken:
                    self._handle_widgets[dut]['button_h'].setDisabled(broken)
                else:
                    if self._handle_widgets[dut]['check_h'].isChecked():
                        self._handle_widgets[dut]['button_h'].setDisabled(broken)
                    else:
                        self._handle_widgets[dut]['button_h'].setDisabled(not broken)

            else:
                if dut in broken_input:
                    self._handle_widgets[dut]['button_h'].setDisabled(True)
                else:
                    if self._handle_widgets[dut]['check_h'].isChecked():
                        self._handle_widgets[dut]['button_h'].setDisabled(False)
                    else:
                        self._handle_widgets[dut]['button_h'].setDisabled(True)

        # Set the status of the proceed button
        if skip_props is None:
            self.btn_ok.setDisabled(broken)

    def _emit_message(self, message):
        """
        Emits statusMessage signal with message
        """

        self.statusMessage.emit(message)

    def _disable_tabs(self):

        for dut in self.data['dut_names']:
            self.tw[dut].setDisabled(True)

        self.btn_ok.setDisabled(True)


class SetupPainter(QtWidgets.QGraphicsView):

    def __init__(self, dut_names, parent=None, plane_width=10, plane_height=100, n_painters=2, rotation_factor=10):
        super(SetupPainter, self).__init__(parent)

        if parent is not None:
            self.w = parent.width()
            self.h = parent.height()/n_painters
        else:
            self.w = 400
            self.h = self.w/n_painters

        self.setMinimumSize(self.w, self.h)

        # Add graphics scene to view
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSceneRect(0, 0, self.w, self.h)
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        # Add stuff
        self.dut_names = dut_names
        self.plane_w = plane_width
        self.plane_h = plane_height
        self.planes = {}
        self.names = {}
        self.z_positions = {}
        self.rotations = {}
        self.mRad = 0.001 * 180.0/3.14159265
        self.unit = None
        self.rot_factor = rotation_factor

        # Get coordinates of scene
        self.left = 0
        self.right = self.w
        self.top = 0
        self.bottom = self.h
        self.h_center = self.w/2
        self.v_center = self.h/2

        # Add different QPens etc. for drawing duts and beam
        self.pen_dut = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)
        self.brush_dut = QtGui.QBrush(QtCore.Qt.SolidPattern)
        self.brush_dut.setColor(QtGui.QColor('lightgrey'))
        self.pen_beam = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.DashLine)
        self.font_plot = QtGui.QFont()
        self.font_plot.setBold(True)
        self.font_z = QtGui.QFont()
        self.font_z.setPointSize(11 - len(self.dut_names)/2)

        self.setToolTip('Rotations shown are enlarged by a factor %d' % self.rot_factor)

        self.setToolTipDuration(5000)

        self._draw_beam()
        self._draw_setup()

    def _draw_setup(self):

        single_dut = False

        # Create dynamic distance in between planes
        try:
            self.dist = self.w / (len(self.dut_names) - 1)

        # Only single DUT
        except ZeroDivisionError:
            self.dist = self.w / (len(self.dut_names) + 1)
            single_dut = True

        for i, dut in enumerate(self.dut_names):

            # Factor for initial equidistant separation of DUTs
            if single_dut:
                factor = self.dist
            else:
                factor = i * self.dist

            plane = QtWidgets.QGraphicsRectItem(factor, self.v_center-self.plane_h*0.5, self.plane_w, self.plane_h)
            plane.setPen(self.pen_dut)
            plane.setBrush(self.brush_dut)
            plane.setTransformOriginPoint(factor, self.v_center)
            name = QtWidgets.QGraphicsTextItem(dut)
            name.setPos(factor - len(dut) * 3, self.v_center / 2 - 40)

            # Add dut plane to dict for changing rotation in update_setup
            self.planes[dut] = plane
            self.names[dut] = name
            self.z_positions[dut] = factor

#            if i == 0:
#
#                z_pos = QtWidgets.QGraphicsTextItem('0 ' + u'\u03BC' + 'm')
#                z_pos.setPos(factor * self.dist - len(dut) * 5, self.v_center+2*self.v_center/3)
#                z_pos.setFont(self.font_z)
#                self.z_positions[dut] = z_pos
#                self.scene.addItem(self.z_positions[dut])

            self.scene.addItem(self.planes[dut])
            self.scene.addItem(name)

    def _draw_beam(self, arrow_length=10, offset=[90, 15]):

        # Draw beam axis in vertical center of scene
        self.scene.addLine(self.left-offset[0]/3, self.v_center, self.right+offset[0]/3, self.v_center, self.pen_beam)

        # Draw arrow lines
        self.scene.addLine(self.right+offset[0]/3 - arrow_length, self.v_center + arrow_length,
                           self.right+offset[0]/3, self.v_center, self.pen_dut)
        self.scene.addLine(self.right+offset[0]/3 - arrow_length, self.v_center - arrow_length,
                           self.right+offset[0]/3, self.v_center, self.pen_dut)

        # Add beam label
        label_beam = QtWidgets.QGraphicsTextItem('Beam')
        label_beam.setFont(self.font_plot)
        label_beam.setPos(self.left-offset[0], self.v_center - offset[1])
        self.scene.addItem(label_beam)

    def draw_coordinate_system(self, origin=None, axes=['x', 'y', 'z'], axis_length=40, arrow_length=4, offset=[80,20]):

        if origin is None:
            origin = (-offset[0], self.h + offset[1])

        pen_cs = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)

        # Make axis labels; 0==x, 1==y, 2==z
        labels = {}

        for i in range(len(axes)):
            labels[i] = QtWidgets.QGraphicsTextItem(axes[i])

            # X axis
            if i == 0:
                self.scene.addLine(origin[0], origin[1], origin[0] + axis_length, origin[1], pen_cs)

                self.scene.addLine(origin[0] + axis_length - arrow_length, origin[1] - arrow_length,
                                   origin[0] + axis_length, origin[1], pen_cs)

                self.scene.addLine(origin[0] + axis_length - arrow_length, origin[1] + arrow_length,
                                   origin[0] + axis_length, origin[1], pen_cs)

                labels[i].setPos(origin[0] + axis_length, origin[1] - 2 * axis_length/10)
                self.scene.addItem(labels[i])

            # Y axis
            if i == 1:
                self.scene.addLine(origin[0], origin[1], origin[0], origin[1] - axis_length, pen_cs)

                self.scene.addLine(origin[0] - arrow_length, origin[1] - axis_length + arrow_length,
                                   origin[0], origin[1] - axis_length, pen_cs)

                self.scene.addLine(origin[0] + arrow_length, origin[1] - axis_length + arrow_length,
                                   origin[0], origin[1] - axis_length, pen_cs)

                labels[i].setPos(origin[0] - 3 * arrow_length, origin[1] - 1.5 * axis_length)
                self.scene.addItem(labels[i])

            # Z axis
            if i == 2:

                factor_1 = 1/2**0.5
                factor_2 = 2**0.5

                self.scene.addLine(origin[0], origin[1],
                                   origin[0] + (axis_length * factor_1), origin[1] - (axis_length * factor_1),
                                   pen_cs)

                self.scene.addLine(origin[0] + (axis_length * factor_1),
                                   origin[1] - (axis_length * factor_1) + (arrow_length * factor_2),
                                   origin[0] + (axis_length * factor_1),
                                   origin[1] - (axis_length * factor_1), pen_cs)

                self.scene.addLine(origin[0] + (axis_length * factor_1) - (arrow_length * factor_2),
                                   origin[1] - (axis_length * factor_1),
                                   origin[0] + (axis_length * factor_1),
                                   origin[1] - (axis_length * factor_1), pen_cs)

                labels[i].setPos(origin[0] + (axis_length * factor_1),
                                 origin[1] - (axis_length * factor_1) - 4 * axis_length/10)
                self.scene.addItem(labels[i])

    def set_rotation(self, dut, rotation=None):

        if rotation is not None:

            try:
                rot = float(rotation)
            except ValueError:
                rot = 0

            self.rotations[dut] = rot

            self.planes[dut].setRotation(self.rotations[dut] * self.mRad * self.rot_factor)

        else:

            for dut_1 in self.planes.keys():
                try:
                    self.planes[dut_1].setRotation(self.rotations[dut_1]*self.mRad * self.rot_factor)
                except KeyError:
                    pass

    def set_z_pos(self, dut, z_position):

        try:
            z_pos = float(z_position)
        except (TypeError, ValueError):
            z_pos = 0

        self.z_positions[dut] = z_pos

        if dut == self.dut_names[-1]:

            try:
                self.unit = self.w / self.z_positions[self.dut_names[-1]]
            except ZeroDivisionError:
                self.unit = 0

            for dut_1 in self.planes.keys():

                factor = self.z_positions[dut_1] * self.unit

                self.scene.removeItem(self.planes[dut_1])

                plane = QtWidgets.QGraphicsRectItem(factor, self.v_center-self.plane_h*0.5, self.plane_w, self.plane_h)
                plane.setPen(self.pen_dut)
                plane.setBrush(self.brush_dut)
                plane.setTransformOriginPoint(factor, self.v_center)

                self.planes[dut_1] = plane
                self.names[dut_1].setPos(factor - len(dut_1) * 3, self.v_center / 2 - 40)

                self.scene.addItem(self.planes[dut_1])

        if self.unit is not None:

                factor = z_pos * self.unit

                self.scene.removeItem(self.planes[dut])

                plane = QtWidgets.QGraphicsRectItem(factor, self.v_center - self.plane_h * 0.5,
                                                    self.plane_w, self.plane_h)
                plane.setPen(self.pen_dut)
                plane.setBrush(self.brush_dut)
                plane.setTransformOriginPoint(factor, self.v_center)

                self.planes[dut] = plane
                self.names[dut].setPos(factor - len(dut) * 3, self.v_center / 2 - 40)

                self.scene.addItem(self.planes[dut])
