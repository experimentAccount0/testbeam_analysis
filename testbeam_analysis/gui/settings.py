from PyQt5 import QtCore, QtWidgets, QtGui
from copy import deepcopy


class SettingsWindow(QtWidgets.QMainWindow):

    settingsUpdated = QtCore.pyqtSignal()

    def __init__(self, setup=None, options=None, parent=None):
        """
        Create window to set global settings for analysis
        """

        super(SettingsWindow, self).__init__(parent)

        self.window_title = 'Global settings'

        self.default_setup = {'n_duts': 3,
                              'n_pixels': [(10, 10)],
                              'pixel_size': [(100, 100)],
                              'dut_name': 'myDUT',
                              'z_positions': (0, 10000),
                              'dut_names': ('First DUT', '2nd DUT', '3rd DUT')}

        self.default_options = {'input_files': '', 'output_path': '', 'input_hits_file': 'test_DUT0.h5',
                                'output_mask_file': 'tt', 'chunk_size': 1000000, 'plot': False,
                                'input_cluster_files': 'test.h5', 'output_correlation_file': 'tt.h5',
                                'noisy_suffix': '_noisy.h5', 'cluster_suffix': '_clustered.h5'}

        # Make copy of defaults to change values but dont change defaults
        if setup is None:
            self.setup = deepcopy(self.default_setup)
        else:
            self.setup = setup

        if options is None:
            self.options = deepcopy(self.default_options)
        else:
            self.options = options

        self._init_UI()

    def _init_UI(self):
        """
        Create user interface
        """

        # Settings window
        self.setWindowTitle(self.window_title)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(0.5 * self.screen.width(), 0.5 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Widgets and layout
        # Spacing related
        v_space = 30
        h_space = 15

        # Make central widget
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(v_space)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Make QGridLayout for options
        layout_options = QtWidgets.QGridLayout()
        layout_options.setSpacing(h_space)

        # Make widgets for plot option
        label_plot = QtWidgets.QLabel('Plot:')
        self.rb_t = QtWidgets.QRadioButton('True')
        self.rb_f = QtWidgets.QRadioButton('False')
        if self.options['plot']:
            self.rb_t.setChecked(True)
        else:
            self.rb_f.setChecked(True)

        # Make widgets for chunk size option
        label_chunk = QtWidgets.QLabel('Chunk size:')
        self.edit_chunk = QtWidgets.QLineEdit()
        valid_chunk = QtGui.QIntValidator()
        valid_chunk.setBottom(0)
        self.edit_chunk.setValidator(valid_chunk)
        self.edit_chunk.setText(str(self.options['chunk_size']))

        # Make widgets for input file suffixes
        label_suffix = QtWidgets.QLabel('Input file suffixes:')
        self.edit_noisy = QtWidgets.QLineEdit()
        self.edit_noisy.setPlaceholderText('NoisyPixelSuffix = %s' % self.options['noisy_suffix'])
        self.edit_cluster = QtWidgets.QLineEdit()
        self.edit_cluster.setPlaceholderText('ClusterSuffix = %s' % self.options['cluster_suffix'])

        # Add all  option widgets to layout_options, add spacers
        layout_options.addWidget(label_plot, 0, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 0, 1, 1, 1)
        layout_options.addWidget(self.rb_t, 0, 2, 1, 1)
        layout_options.addWidget(self.rb_f, 0, 3, 1, 1)
        layout_options.addWidget(label_chunk, 1, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 1, 1, 1, 1)
        layout_options.addWidget(self.edit_chunk, 1, 2, 1, 2)
        layout_options.addWidget(label_suffix, 2, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 2, 1, 1, 1)
        layout_options.addWidget(self.edit_noisy, 2, 2, 1, 2)
        layout_options.addWidget(self.edit_cluster, 3, 2, 1, 2)

        # Make buttons for apply settings and cancel and button layout
        layout_buttons = QtWidgets.QHBoxLayout()
        button_ok = QtWidgets.QPushButton('Ok')
        button_ok.clicked.connect(lambda: self._update_settings())
        button_cancel = QtWidgets.QPushButton('Cancel')
        button_cancel.clicked.connect(lambda: self.close())
        layout_buttons.addStretch(1)
        layout_buttons.addWidget(button_ok)
        layout_buttons.addWidget(button_cancel)

        # Add all layouts to main layout
        main_layout.addSpacing(v_space)
        main_layout.addLayout(layout_options)
        main_layout.addStretch(1)
        main_layout.addLayout(layout_buttons)

    def _update_settings(self):

        palette = QtGui.QPalette()

        try:
            n = int(self.edit_chunk.text())
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit_chunk.setPalette(palette)
            if self.options['chunk_size'] != n:
                self.options['chunk_size'] = n

        except (TypeError, ValueError):
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit_chunk.setPalette(palette)
            n = self.edit_chunk.text()
            self.statusBar().showMessage('Chunk size must be an integer, is type %s !' % type(n), 2000)
            return

        self.options['plot'] = self.rb_t.isChecked()
        self.settingsUpdated.emit()
        self.close()
