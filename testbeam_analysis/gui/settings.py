from PyQt5 import QtCore, QtWidgets, QtGui

window_title = 'Global settings'


class DefaultSettings(object):

    def __init__(self):
        self.setup = {'n_duts': 2,
                      'n_pixels': [(10, 10)],
                      'pixel_size': [(100, 100)],
                      'dut_name': 'myDUT',
                      'z_positions': (0, 10000),
                      'dut_names': ('First DUT', 'SCD DUT')}

        self.options = {'working_directory': '',
                        'input_hits_file': 'test_DUT0.h5',
                        'output_mask_file': 'tt',
                        'chunk_size': 1000000,
                        'plot': False}

        self.options['input_cluster_files'] = 'test.h5'
        self.options['output_correlation_file'] = 'tt.h5'


class SettingsWindow(QtWidgets.QMainWindow):

    settingsUpdated = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        """
        Create window to set global settings for analysis such as n_pixels,
        output_folder etc.
        """

        super(SettingsWindow, self).__init__(parent)
        ds = DefaultSettings()
        self.setup = ds.setup
        self.options = ds.options
        self._init_UI()

    def _init_UI(self):
        """
        Create user interface
        """

        # Settings window
        self.setWindowTitle(window_title)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(0.25 * self.screen.width(), 0.25 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Widgets and layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(10)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        layout_plot = QtWidgets.QHBoxLayout()
        label_plot = QtWidgets.QLabel('Plot:')
        self.rb_t = QtWidgets.QRadioButton('True')
        self.rb_f = QtWidgets.QRadioButton('False')
        if self.options['plot']:
            self.rb_t.setChecked(True)
        else:
            self.rb_f.setChecked(True)
        layout_plot.addWidget(label_plot)
        layout_plot.setSpacing(10)
        layout_plot.addWidget(self.rb_t)
        layout_plot.addWidget(self.rb_f)
        layout_chunk = QtWidgets.QHBoxLayout()
        label_chunk = QtWidgets.QLabel('Chunk size:')
        self.edit_chunk = QtWidgets.QLineEdit()
        self.edit_chunk.setText(str(self.options['chunk_size']))
        self.edit_chunk.setFixedWidth(0.5 * self.width())
        layout_chunk.addWidget(label_chunk)
        layout_chunk.setSpacing(10)
        layout_chunk.addWidget(self.edit_chunk)
        button_update = QtWidgets.QPushButton('Update settings')
        button_update.clicked.connect(lambda: self._update_settings())
        main_layout.addLayout(layout_plot)
        main_layout.addLayout(layout_chunk)
        main_layout.addWidget(button_update)

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

        if self.rb_t.isChecked():
            self.options['plot'] = True
        else:
            self.options['plot'] = False

        self.statusBar().showMessage('Settings updated!', 2000)
        self.settingsUpdated.emit()
