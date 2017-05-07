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

        # Make list of tabs that will be enabled after proceedAnalysis signal of this class
        self.tab_list = ['Noisy Pixel', 'Clustering']

        # Make tuple of properties of each dut
        self._dut_props = ('z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma',
                           'pitch_col', 'pitch_row', 'n_cols', 'n_rows', 'thickness')

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
        label_setup = QtWidgets.QLabel('Schematic setup:')
        self.draw.addWidget(label_setup)
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
        self.button_ok = QtWidgets.QPushButton('OK')
        self.button_ok.clicked.connect(lambda: self._handle_input())
        self.button_ok.setDisabled(True)
        layout_right.addWidget(self.button_ok)
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

        self.setup_painter = SetupPainter(dut_names, self.left_widget)

        self.draw.addWidget(self.setup_painter)

    def _init_tabs(self):
        """
        Initializes the setup tabs and their widget contents for each dut in the dut_names list from DataTab
        """

        # Clear the QTabwidget which holds each duts tab to allow updating the data from DataTab
        self.tabs.clear()

        # Spacing related numbers
        h_space = 10
        v_space = 30
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
#            checkbox_type = QtWidgets.QCheckBox()
#            checkbox_type.setText('Set DUT type')
#            checkbox_type.setDisabled(True)
#            checkbox_type.setFixedWidth(label_width)
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
            edit_alpha.setPlaceholderText(u'\u03B1' + ' = 0')
            edit_beta = QtWidgets.QLineEdit()
            edit_beta.setPlaceholderText(u'\u03B2' + ' = 0')
            edit_gamma = QtWidgets.QLineEdit()
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
            label_thickness.setFixedWidth(label_width)
            edit_thickness = QtWidgets.QLineEdit()

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
#            layout_props.addWidget(checkbox_type, 0, 0, 1, 1)
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
            layout_props.addWidget(checkbox_handle, 4, 0, 1, 1)
            layout_props.addItem(QtWidgets.QSpacerItem(h_space, v_space), 4, 1, 1, 1)
            layout_props.addWidget(edit_handle, 4, 2, 1, 1)
            layout_props.addWidget(button_handle, 4, 3, 1, 1)
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
                            edit_pitch_row, edit_pixels_col, edit_pixels_row, edit_thickness)

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
#            self._type_widgets[dut] = {'check_t': checkbox_type, 'combo_t': cb_type, 'button_t': button_type}
            self._type_widgets[dut] = {'combo_t': cb_type, 'button_t': button_type}

            # Add all dut type creating related widgets to a dict with respective dut as key
            self._handle_widgets[dut] = {'check_h': checkbox_handle, 'edit_h': edit_handle, 'button_h': button_handle}

            # Add all button widgets to a dict with respective dut as key
            self._buttons[dut] = {'global': button_glob, 'clear': button_clear}

            # Add tab to QTabWidget
            self.tabs.addTab(widget, dut)

        # Connect tabs
        # self.tabs.currentChanged.connect(lambda tab: self._handle_dut_types(self.data['dut_names'][tab])) # FIXME

        # Connect widgets of all tabs
        for dut in self.data['dut_names']:

            # Connect set type button
            self._type_widgets[dut]['button_t'].clicked.connect(
                lambda: self._set_properties(
                    self.data['dut_names'][self.tabs.currentIndex()],
                    self._type_widgets[self.data['dut_names'][self.tabs.currentIndex()]]['combo_t'].currentText()))

#            # Connect set type checkbox
#            self._type_widgets[dut]['check_t'].toggled.connect(
#                lambda: self._handle_dut_types(self.data['dut_names'][self.tabs.currentIndex()]))

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

                if prop in ['z_positions', 'rot_alpha', 'rot_beta', 'rot_gamma']:
                    self._dut_widgets[dut][prop].textChanged.connect(
                        lambda text: self.setup_painter.update_setup(
                            self.data['dut_names'][self.tabs.currentIndex()], self.tabs.currentIndex(), text))

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
                    elif prop in ['rot_alpha', 'rot_beta', 'rot_gamma']:
                        # Default value for rotations is 0
                        if not self._dut_widgets[dut][prop].text():
                            tmp_rotation[prop] = 0.0
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
                            self._dut_types[custom][prop] = float(self._dut_widgets[current_dut][prop].text())

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

                #  if self._type_widgets[dut]['check_t'].isChecked():
                    #  for key in self._type_widgets[dut].keys():
                        #  self._type_widgets[dut][key].setDisabled(False)
                #  else:
                    #  for key in self._type_widgets[dut].keys():
                        #  if key is not 'check_t':
                            #  self._type_widgets[dut][key].setDisabled(True)

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

                if prop in ['rot_alpha', 'rot_beta', 'rot_gamma']: #FIXME: test-wise excluding rotations from checking
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

                self._handle_widgets[dut]['button_h'].setDisabled(broken)

            else:
                if dut in broken_input:
                    self._handle_widgets[dut]['button_h'].setDisabled(True)
                else:
                    self._handle_widgets[dut]['button_h'].setDisabled(False)

        # Set the status of the proceed button
        if skip_props is None:
            self.button_ok.setDisabled(broken)

    def _emit_message(self, message):
        """
        Emits statusMessage signal with message
        """

        self.statusMessage.emit(message)


class SetupPainter(QtWidgets.QGraphicsView):

    def __init__(self, dut_names, parent=None):
        super(SetupPainter, self).__init__(parent)

        # Handle appearance of view
        self.setGeometry(parent.geometry())
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        width = self.frameSize().width() * (1 - (self.frameSize().width() / 2) * (1 / (len(dut_names)+1)))
        self.setSceneRect(0, 0, width, self.frameSize().height())
        self.setRenderHint(QtGui.QPainter.Antialiasing)

        # Add dicts to save duts z positions and rotation values
        self.planes = {}
        self.rotations = {}
        self.z_positions = {}

        # Get coordinates of scene
        self.left = 0
        self.right = width
        self.top = 0
        self.bottom = self.frameSize().height()
        self.h_center = width / 2
        self.v_center = self.frameSize().height() / 2

        # Add different QPens etc. for drawing duts and beam
        pen_dut = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)
        brush_dut = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush_dut.setColor(QtGui.QColor('lightgrey'))
        pen_beam = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.DashLine)
        font_plot = QtGui.QFont()
        font_plot.setBold(True)

        # Create dynamic distance in between planes
        self.dist = width / (len(dut_names) + 1)

        # Draw beam axis in vertical center of scene, add labels
        self.scene.addLine(self.left, self.v_center, self.right, self.v_center, pen_beam)
        self.scene.addLine(self.right-10, self.v_center+10, self.right, self.v_center, pen_dut)
        self.scene.addLine(self.right-10, self.v_center-10, self.right, self.v_center, pen_dut)
        label_beam = QtWidgets.QGraphicsTextItem('Beam')
        label_beam.setFont(font_plot)
        label_beam.setPos(-60, self.v_center-15)
        label_axis = QtWidgets.QGraphicsTextItem('z-Axis')
        label_axis.setFont(font_plot)
        label_axis.setPos(self.right, self.v_center-15)
        self.scene.addItem(label_beam)
        # self.scene.addItem(label_axis)

        # Draw coordinate system in lower left corner # FIXME: Hardcoded
        # Draw axes
        # X
        self.scene.addLine(self.left, self.bottom, self.left, self.bottom-50, pen_dut)
        # Y
        self.scene.addLine(self.left, self.bottom, self.left + (50 * 0.707), self.bottom - (50 * 0.707), pen_dut)
        # Z
        self.scene.addLine(self.left, self.bottom, self.left+50, self.bottom, pen_dut)
        # Make arrowheads for axes
        # X
        self.scene.addLine(self.left-5, self.bottom-45, self.left, self.bottom - 50, pen_dut)
        self.scene.addLine(self.left + 5, self.bottom - 45, self.left, self.bottom - 50, pen_dut)
        # Y
        self.scene.addLine(self.left + (50 * 0.707), self.bottom - (50 * 0.707) + (5 * 1.41), self.left + (50 * 0.707),
                           self.bottom - (50 * 0.707), pen_dut)
        self.scene.addLine(self.left + (50 * 0.707) - (5 * 1.41), self.bottom - (50 * 0.707), self.left + (50 * 0.707),
                           self.bottom - (50 * 0.707), pen_dut)
        # Z
        self.scene.addLine(self.left + 45, self.bottom - 5, self.left + 50, self.bottom, pen_dut)
        self.scene.addLine(self.left + 45, self.bottom + 5, self.left + 50, self.bottom, pen_dut)
        # Make labels for axis
        label_x = QtWidgets.QGraphicsTextItem('x')
        label_x.setPos(self.left-10, self.bottom-75)
        label_y = QtWidgets.QGraphicsTextItem('y')
        label_y.setPos(self.left + (50*0.707), self.bottom-(50*0.707) - 20)
        label_z = QtWidgets.QGraphicsTextItem('z')
        label_z.setPos(self.left + 50, self.bottom - 10)
        # Add axes labels
        self.scene.addItem(label_x)
        self.scene.addItem(label_y)
        self.scene.addItem(label_z)

        for i, dut in enumerate(dut_names):

            # Factor with which dist is multiplied
            factor = i+1

            # Initialize the dicts with None for each dut
            self.z_positions[dut] = None
            self.rotations[dut] = {'alpha': None, 'beta': None, 'gamma': None}

            plane = QtWidgets.QGraphicsRectItem(factor * self.dist, self.v_center/2, 20, self.v_center)
            plane.setPen(pen_dut)
            plane.setBrush(brush_dut)
            plane.setTransformOriginPoint(factor * self.dist, self.v_center)

            # Add dut plane to dict for changing rotation in update_setup
            self.planes[dut] = plane

            name = QtWidgets.QGraphicsTextItem(dut)
            name.setPos(factor * self.dist - len(dut)*3, self.v_center / 2 - 40)

            if i == 0:

                z_pos = QtWidgets.QGraphicsTextItem('z: 0 ' + u'\u03BC' + 'm')
                z_pos.setPos(factor * self.dist - len(dut) * 5, self.v_center+2*self.v_center/3)
                self.scene.addItem(z_pos)

            self.scene.addItem(self.planes[dut])
            self.scene.addItem(name)

    def update_setup(self, dut, index, rotation=None):
        return # TODO
        i = index+1
        for j in range(len(self.rotations[dut].keys())):
            for key in self.rotations[dut].keys():
                a = QtWidgets.QGraphicsTextItem(u'\u03B1' + ': %s' % self.rotations[dut][key] + ' mRad')
                a.setPos(i * self.dist - len(dut) * 5, self.v_center+150+j*40)
            self.scene.addItem(a)
