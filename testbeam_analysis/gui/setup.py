import os

from PyQt5 import QtCore, QtWidgets, QtGui


class SetupTab(QtWidgets.QWidget):
    """
    Implements the tab content for data file handling
    """

    statusMessage = QtCore.pyqtSignal(['QString'])
    proceedAnalysis = QtCore.pyqtSignal()

    def __init__(self, parent=None, input_files=None, dut_names=None):
        super(SetupTab, self).__init__(parent)

        # Add output data
        self.data = None
        self.dut_data = {}
        self._dut_specs = ('z-Position', 'Rotation', 'Pixel pitch', 'Number of pixels', 'Thickness')
        self._dut_widgets = {}

        self._setup()

    def get_data(self, data):
        self.data = data

        self._update_tabs()

    def _setup(self):
        # Plot area
        left_widget = QtWidgets.QWidget()
        self.draw = QtWidgets.QHBoxLayout()
        l = QtWidgets.QLabel('Wow. Great generic plotting of telescope!')
        self.draw.addWidget(l)
        left_widget.setLayout(self.draw)
        # Dut setup
        layout_tabs = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        layout_tabs.addWidget(self.tabs)
        layout_options = QtWidgets.QVBoxLayout()
        layout_options.addLayout(layout_tabs)
        # Proceed button
        button_ok = QtWidgets.QPushButton('OK')
        layout_options.addWidget(button_ok)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(layout_options)
        # Split plot and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 5)
        widget_splitter.setStretchFactor(1, 10)
        widget_splitter.setChildrenCollapsible(False)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def _update_tabs(self):
        self.tabs.clear()
        for name in self.data['dut_names']:
            for spec in self._dut_specs:
                self._dut_widgets[spec] = QtWidgets.QLineEdit()
            widget = QtWidgets.QWidget()
            self.tabs.addTab(widget, name)