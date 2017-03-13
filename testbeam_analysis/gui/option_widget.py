''' Implements option widgets to set function arguments.

    Options can set numbers, strings and booleans.
    Optional options can be deactivated with the value None.
'''

from PyQt5 import QtWidgets, QtCore, QtGui


class OptionSlider(QtWidgets.QWidget):  # FIXME: steps size != 1 not supported
    ''' Option slider for numbers

        Shows the value as text and can increase range
    '''

    valueChanged = QtCore.pyqtSignal([float])

    def __init__(self, name, default_value, optional, tooltip, parent=None):
        super(OptionSlider, self).__init__(parent)

        # Slider with textbox to the right
        layout_2 = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.edit = QtWidgets.QLineEdit()
        validator = QtGui.QDoubleValidator()
        self.edit.setValidator(validator)
        self.edit.setMaxLength(3)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                            QtWidgets.QSizePolicy.Preferred)
        self.edit.setSizePolicy(size_policy)
        layout_2.addWidget(slider)
        layout_2.addWidget(self.edit)

        # Option name with slider below
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(name)
        if tooltip:
            text.setToolTip(tooltip)
        if optional:
            layout_1 = QtWidgets.QHBoxLayout()
            layout_1.addWidget(text)
            layout_1.addStretch(0)
            check_box = QtWidgets.QCheckBox()
            layout_1.addWidget(check_box)
            layout.addLayout(layout_1)

            check_box.stateChanged.connect(
                lambda v: self._set_readonly(v == 0))
            self._set_readonly()
        else:
            layout.addWidget(text)
        layout.addLayout(layout_2)

        slider.valueChanged.connect(lambda v: self.edit.setText(str(v)))
        self.edit.returnPressed.connect(
            lambda: slider.setMaximum(float(self.edit.text()) * 2))
        slider.valueChanged.connect(lambda _: self._emit_value())

        if default_value is not None:
            slider.setValue(default_value)
            # Needed because set value does not issue a value changed
            # if value stays constant
            self.edit.setText(str(slider.value()))

    def _set_readonly(self, value=True):
        if value:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(True)
            self._emit_value()
        else:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(False)
            self._emit_value()

    def _emit_value(self):
        if self.edit.isReadOnly() or not self.edit.text():
            self.valueChanged.emit(float('nan'))
        else:
            self.valueChanged.emit(int(self.edit.text()))


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

        self.edit.textChanged.connect(lambda: self._emit_value())

        if default_value is not None:
            self.edit.setText(default_value)

    def _set_readonly(self, value=True):
        if value:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(True)
            self._emit_value()
        else:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(False)
            self._emit_value()

    def _emit_value(self):
        if self.edit.isReadOnly():
            self.valueChanged.emit('None')
        else:
            self.valueChanged.emit(self.edit.text())


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

        rb_t.toggled.connect(lambda: self.valueChanged.emit(rb_t.isChecked()))

        if tooltip:
            text.setToolTip(tooltip)

        if default_value is not None:
            rb_t.setChecked(default_value is True)
            rb_f.setChecked(default_value is False)


class OptionMultiSlider(QtWidgets.QWidget):  # FIXME: steps size != 1 not supported
    ''' Option sliders for several numbers

        Shows the value as text and can increase range
    '''

    valueChanged = QtCore.pyqtSignal([float])

    def __init__(self, name, n_values, default_value, optional, tooltip, parent=None):
        super(OptionMultiSlider, self).__init__(parent)

        # Slider with textbox to the right
        layout_2 = QtWidgets.QHBoxLayout()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.edit = QtWidgets.QLineEdit()
        validator = QtGui.QDoubleValidator()
        self.edit.setValidator(validator)
        self.edit.setMaxLength(3)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                            QtWidgets.QSizePolicy.Preferred)
        self.edit.setSizePolicy(size_policy)
        layout_2.addWidget(slider)
        layout_2.addWidget(self.edit)

        # Option name with slider below
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(name)
        if tooltip:
            text.setToolTip(tooltip)
        if optional:
            layout_1 = QtWidgets.QHBoxLayout()
            layout_1.addWidget(text)
            layout_1.addStretch(0)
            check_box = QtWidgets.QCheckBox()
            layout_1.addWidget(check_box)
            layout.addLayout(layout_1)

            check_box.stateChanged.connect(
                lambda v: self._set_readonly(v == 0))
            self._set_readonly()
        else:
            layout.addWidget(text)
        layout.addLayout(layout_2)

        slider.valueChanged.connect(lambda v: self.edit.setText(str(v)))
        self.edit.returnPressed.connect(
            lambda: slider.setMaximum(float(self.edit.text()) * 2))
        slider.valueChanged.connect(lambda _: self._emit_value())

        if default_value is not None:
            slider.setValue(default_value)
            # Needed because set value does not issue a value changed
            # if value stays constant
            self.edit.setText(str(slider.value()))

    def _set_readonly(self, value=True):
        if value:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(True)
            self._emit_value()
        else:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit.setPalette(palette)
            self.edit.setReadOnly(False)
            self._emit_value()

    def _emit_value(self):
        if self.edit.isReadOnly() or not self.edit.text():
            self.valueChanged.emit(float('nan'))
        else:
            self.valueChanged.emit(int(self.edit.text()))
