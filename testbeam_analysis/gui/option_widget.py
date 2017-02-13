''' Implements option widgets to set function arguments
'''

from PyQt5 import QtWidgets, QtCore, QtGui


class OptionSlider(QtWidgets.QWidget):  # FIXME: optional not implemented
    ''' Option slider for numbers

        Shows the value as text and can increase range
    '''

    valueChanged = QtCore.pyqtSignal([int], [float])

    def __init__(self, name, default_value, optional, tooltip, parent=None):
        super(OptionSlider, self).__init__(parent)

        # Slider with textbox to the right
        layout_2 = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        value = QtWidgets.QLineEdit()
        validator = QtGui.QDoubleValidator()
        value.setValidator(validator)
        value.setMaxLength(3)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                            QtWidgets.QSizePolicy.Preferred)
        value.setSizePolicy(size_policy)
        layout_2.addWidget(slider)
        layout_2.addWidget(value)

        # Option name with slider below
        layout_1 = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(name)
        if tooltip:
            text.setToolTip(tooltip)
        layout_1.addWidget(text)
        layout_1.addLayout(layout_2)

        slider.valueChanged.connect(lambda v: value.setText(str(v)))
        value.returnPressed.connect(
            lambda: slider.setMaximum(float(value.text()) * 2))
        slider.valueChanged.connect(lambda v: self.valueChanged.emit(v))

        if default_value is not None:
            slider.setValue(default_value)
            # Needed because set value does not issue a value changed
            # if value stays constant
            value.setText(str(slider.value()))


class OptionText(QtWidgets.QWidget):
    ''' Option text for strings
    '''

    valueChanged = QtCore.pyqtSignal(['QString'])

    def __init__(self, name, default_value, optional, tooltip=None, parent=None):
        super(OptionText, self).__init__(parent)
        self.edit = QtWidgets.QLineEdit()
        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QLabel(name)
        if optional:
            layout_1 = QtWidgets.QHBoxLayout()
            layout_1.addWidget(text)
            layout_1.addStretch(0)
            check_box = QtWidgets.QCheckBox()
            layout_1.addWidget(check_box)
            layout.addLayout(layout_1)

            check_box.stateChanged.connect(
                lambda v: self._set_readonly(v == 0))
            if not default_value:
                check_box.setCheckState(0)
                self._set_readonly(True)
        else:
            layout.addWidget(text)

        if tooltip:
            text.setToolTip(tooltip)

        layout.addWidget(self.edit)

        self.edit.textChanged.connect(
            lambda: self.valueChanged.emit(self.edit.text()))

        if default_value is not None:
            self.edit.setText(default_value)

    def _set_readonly(self, value=True):
        if value:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(True)
        else:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(False)


class OptionBool(QtWidgets.QWidget):  # FIXME: optional not implemented
    ''' Option bool for booleans
    '''

    valueChanged = QtCore.pyqtSignal(bool)

    def __init__(self, name, default_value, optional, tooltip=None, parent=None):
        super(OptionBool, self).__init__(parent)
        rb_t = QtWidgets.QRadioButton('True')
        rb_f = QtWidgets.QRadioButton('False')
        layout_b = QtWidgets.QHBoxLayout()
        layout_b.addWidget(rb_t)
        layout_b.addWidget(rb_f)

        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QLabel(name)
        layout.addWidget(text)
        layout.addLayout(layout_b)

        rb_t.toggled.connect(lambda v: self.valueChanged.emit(rb_t.isChecked()))

        if tooltip:
            text.setToolTip(tooltip)

        if default_value is not None:
            rb_t.setChecked(default_value is True)
            rb_f.setChecked(default_value is False)

    def _set_readonly(self, value=True):
        if value:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(True)
        else:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(False)
