import yaml
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

        self.default_setup = {'dut_names': None,
                              'n_duts': None,
                              'n_pixels': None,
                              'pixel_size': None,
                              'z_positions': None,
                              'rotations': None,
                              'scatter_planes': None}

        self.default_options = {'input_files': None,
                                'output_path': None,
                                'chunk_size': 1000000,
                                'plot': False,
                                'noisy_suffix': '_noisy.h5',  # fixed since fixed in function
                                'cluster_suffix': '_clustered.h5',  # fixed since fixed in function
                                'skip_alignment': False}

        # Make copy of defaults to change values but don't change defaults
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
        self.resize(0.25 * self.screen.width(), 0.25 * self.screen.height())
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
        self.group_plot = QtWidgets.QButtonGroup()
        self.group_plot.addButton(self.rb_t)
        self.group_plot.addButton(self.rb_f)

        if self.options['plot']:
            self.rb_t.setChecked(True)
        else:
            self.rb_f.setChecked(True)

        # Make widgets for skip alignment option
        label_align = QtWidgets.QLabel('Skip alignment:')
        self.rb_t_align = QtWidgets.QRadioButton('True')
        self.rb_f_align = QtWidgets.QRadioButton('False')
        self.group_align = QtWidgets.QButtonGroup()
        self.group_align.addButton(self.rb_t_align)
        self.group_align.addButton(self.rb_f_align)

        if self.options['skip_alignment']:
            self.rb_t_align.setChecked(True)
        else:
            self.rb_f_align.setChecked(True)

        # Make widgets for chunk size option
        label_chunk = QtWidgets.QLabel('Chunk size:')
        self.edit_chunk = QtWidgets.QLineEdit()
        valid_chunk = QtGui.QIntValidator()
        valid_chunk.setBottom(0)
        self.edit_chunk.setValidator(valid_chunk)
        self.edit_chunk.setText(str(self.options['chunk_size']))

        # Add all  option widgets to layout_options, add spacers
        layout_options.addWidget(label_plot, 0, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 0, 1, 1, 1)
        layout_options.addWidget(self.rb_t, 0, 2, 1, 1)
        layout_options.addWidget(self.rb_f, 0, 3, 1, 1)
        layout_options.addWidget(label_align, 1, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7 * h_space, v_space), 1, 1, 1, 1)
        layout_options.addWidget(self.rb_t_align, 1, 2, 1, 1)
        layout_options.addWidget(self.rb_f_align, 1, 3, 1, 1)
        layout_options.addWidget(label_chunk, 2, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 2, 1, 1, 1)
        layout_options.addWidget(self.edit_chunk, 2, 2, 1, 2)

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
        self.options['skip_alignment'] = self.rb_t_align.isChecked()

        self.settingsUpdated.emit()
        self.close()


class ExceptionWindow(QtWidgets.QMainWindow):

    exceptionRead = QtCore.pyqtSignal()

    def __init__(self, exception, traceback, tab=None, cause=None, parent=None):

        super(ExceptionWindow, self).__init__(parent)

        # Make this window blocking parent window
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        # Get important information of the exception
        self.exception = exception
        self.traceback = traceback
        self.exc_type = type(self.exception).__name__

        # Make main message and label
        msg = "The following exception occurred during %s: %s.\n" \
              "Try changing the input parameters. %s tab will be reset!" % (cause, self.exc_type, tab)

        self.label = QtWidgets.QLabel(msg)
        self.label.setWordWrap(True)

        # Make warning icon via pixmap on QLabel
        self.pix_map = QtWidgets.qApp.style().standardIcon(QtWidgets.qApp.style().SP_MessageBoxWarning).pixmap(40, 40)
        self.label_icon = QtWidgets.QLabel()
        self.label_icon.setPixmap(self.pix_map)
        self.label_icon.setFixedSize(40, 40)

        self._init_UI()

    def _init_UI(self):

        # Exceptions window
        self.setWindowTitle(self.exc_type)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setMinimumSize(0.3 * self.screen.width(), 0.3 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Widgets and layout
        # Spacing related
        v_space = 30
        h_space = 15

        # Make central widget
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Layout for icon and label
        layout_labels = QtWidgets.QHBoxLayout()
        layout_labels.addWidget(self.label_icon)
        layout_labels.addWidget(self.label)

        # Layout for buttons
        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch(1)

        # Textbrowser to display traceback
        browser_traceback = QtWidgets.QTextBrowser()
        browser_traceback.setText(self.traceback)

        # Button to safe traceback to file
        btn_safe = QtWidgets.QPushButton('Save')
        btn_safe.setToolTip('Safe traceback to file')
        btn_safe.clicked.connect(self.safe_traceback)

        # Ok button
        btn_ok = QtWidgets.QPushButton('Ok')
        btn_ok.setToolTip('Reset current tab')
        btn_ok.clicked.connect(self.close)

        # Add buttons to layout
        layout_buttons.addWidget(btn_safe)
        layout_buttons.addSpacing(h_space)
        layout_buttons.addWidget(btn_ok)

        # Dock in which text browser is placed
        browser_dock = QtWidgets.QDockWidget()
        browser_dock.setWidget(browser_traceback)
        browser_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        browser_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        browser_dock.setWindowTitle('Traceback:')

        # Add to main layout
        main_layout.addLayout(layout_labels)
        main_layout.addSpacing(v_space)
        main_layout.addWidget(browser_dock)
        main_layout.addLayout(layout_buttons)

    def safe_traceback(self):

        caption = 'Save traceback to file'
        trcbck_path = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                            caption=caption,
                                                            directory='./',
                                                            filter='*.yaml')[0]
        if trcbck_path:
            with open(trcbck_path, 'w') as f_write:
                yaml.dump(self.traceback, f_write, default_flow_style=False)
        else:
            pass

    def closeEvent(self, QCloseEvent):

        self.exceptionRead.emit()
