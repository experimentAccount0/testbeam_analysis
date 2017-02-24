from PyQt5 import QtCore, QtWidgets

window_title = 'Global settings'

setup = {'n_pixel': (10, 10),
         'pixel_size': (100, 100),
         'dut_name': 'myDUT'}

options = {'working_directory': '',
           'input_hits_file': 'test_DUT0.h5',
           'output_mask_file': 'tt',
           'chunk_size': 1000000,
           'plot': False}


class DefaultSettings(object):

    def __init__(self):
        super(DefaultSettings, self).__init__()

        self.setup = setup
        self.options = options


class SettingsWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        """
        Create window to set global settings for analysis such as n_pixels, output_folder etc.
        """

        super(SettingsWindow, self).__init__(parent)
        self.setup = setup
        self.options = options
        self.output_path = str()
        self._init_UI()

    def _init_UI(self):
        """
        Create user interface
        """

        # Settings window
        self.setWindowTitle(window_title)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(0.5 * self.screen.width(), 0.5 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)
        # Widgets
        widget_global = QtWidgets.QWidget()
        widget_types = QtWidgets.QWidget()
        tabs.addTab(widget_global, 'Global')
        tabs.addTab(widget_types, 'DUT types')
        # options

        com = """
        widget_options = QtWidgets.QWidget()
        self.setCentralWidget(widget_options)
        layout_options = QtWidgets.QVBoxLayout()
        label_options = QtWidgets.QLabel('Options')
        layout_options.addWidget(label_options)

        layout_pix_dim = QtWidgets.QHBoxLayout()
        label_pix_dim = QtWidgets.QLabel('Pixel array:')
        label_col = QtWidgets.QLabel('Columns')
        label_row = QtWidgets.QLabel('Rows')
        edit_col = QtWidgets.QLineEdit()
        edit_row = QtWidgets.QLineEdit()
        edit_col.setText(str(setup['n_pixel'][0]))
        edit_row.setText(str(setup['n_pixel'][1]))
        layout_pix_dim.addWidget(label_pix_dim)
        layout_pix_dim.addWidget(label_col)
        layout_pix_dim.addWidget(edit_col)
        layout_pix_dim.addWidget(label_row)
        layout_pix_dim.addWidget(edit_row)

        layout_options.addLayout(layout_pix_dim)
        widget_options.setLayout(layout_options)
        """

    def _get_output_folder(self):
        caption = 'Select output folder'
        self.output_path = QtWidgets.QFileDialog.getExistingDirectory(parent=self,
                                                                      caption=caption,
                                                                      directory='~/')
        self.edit_output.setText(self.output_path)
        self.edit_output.adjustSize()



