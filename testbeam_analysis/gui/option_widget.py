''' Implements option widgets to set function arguments.

    Options can set numbers, strings and booleans.
    Optional options can be deactivated with the value None.
'''

import collections

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
        self.edit.setAlignment(QtCore.Qt.AlignCenter)
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
        slider.valueChanged.connect(lambda _: self._emit_value())
        self.edit.returnPressed.connect(
            lambda: slider.setMaximum(max((float(self.edit.text()) * 2), 1)))
        self.edit.returnPressed.connect(
            lambda: slider.setValue(float(self.edit.text())))

        if default_value is not None:
            slider.setMaximum(max((default_value * 2, 1)))
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
        self.edit.setAlignment(QtCore.Qt.AlignCenter)
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


# FIXME: steps size != 1 not supported
class OptionMultiSlider(QtWidgets.QWidget):
    ''' Option sliders for several numbers

        Shows the value as text and can increase range
    '''

    valueChanged = QtCore.pyqtSignal(list)

    def __init__(self, name, labels, default_value, optional, tooltip, parent=None):
        super(OptionMultiSlider, self).__init__(parent)
        # Check default value
        if default_value is None:  # None is only supported for all values
            default_value = 0.
        if not isinstance(default_value, collections.Iterable):
            default_value = [default_value] * len(labels)
        if len(labels) != len(default_value):
            raise ValueError(
                'Number of default values does not match number of parameters')

        max_val = max((max(default_value) * 2, 1))

        # Option name with sliders below
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(name)
        if tooltip:
            text.setToolTip(tooltip)
        if optional:  # Values can be unset
            layout_1 = QtWidgets.QHBoxLayout()
            layout_1.addWidget(text)
            layout_1.addStretch(0)
            check_box = QtWidgets.QCheckBox()
            layout_1.addWidget(check_box)
            layout.addLayout(layout_1)
        else:
            layout.addWidget(text)

        self.edits = []

        for i, label in enumerate(labels):  # Create one slider per label
            # Slider with textbox to the right
            layout_label = QtWidgets.QHBoxLayout()
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            # Text edit
            edit = QtWidgets.QLineEdit()
            edit.setAlignment(QtCore.Qt.AlignCenter)
            edit.setValidator(QtGui.QDoubleValidator())
            edit.setMaxLength(3)
            size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                                QtWidgets.QSizePolicy.Preferred)
            edit.setSizePolicy(size_policy)
            layout_label.addWidget(QtWidgets.QLabel('  ' + label))
            layout_label.addWidget(slider)
            layout_label.addWidget(edit)

            # Crazy shit: lambda late binding has to be prevented here
            # http://docs.python-guide.org/en/latest/writing/gotchas/
            slider.valueChanged.connect(
                lambda v, edit=edit: edit.setText(str(v)))
            slider.valueChanged.connect(lambda _: self._emit_value())
            edit.returnPressed.connect(lambda slider=slider, edit=edit: slider.setMaximum(
                max(float(edit.text()) * 2, 1)))
            edit.returnPressed.connect(
                lambda slider=slider, edit=edit: slider.setValue(float(edit.text())))

            slider.setMaximum(max_val)
            slider.setValue(default_value[i])
            # Needed because set value does not issue a value changed
            # if value stays constant
            edit.setText(str(slider.value()))

            self.edits.append(edit)

            layout.addLayout(layout_label)

        if optional:
            check_box.stateChanged.connect(
                lambda v: self._set_readonly(v == 0))
            self._set_readonly()

    def _set_readonly(self, value=True):
        if value:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            for edit in self.edits:
                edit.setPalette(palette)
                edit.setReadOnly(True)
            self._emit_value()
        else:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            for edit in self.edits:
                edit.setPalette(palette)
                edit.setReadOnly(False)
            self._emit_value()

    def _emit_value(self):
        if not any([edit.isReadOnly() for edit in self.edits]):
            values = [int(edit.text()) for edit in self.edits]
        else:
            values = [None]
        self.valueChanged.emit(values)


class OptionMultiBox(QtWidgets.QWidget):
    ''' Option boxes 2(NxN) dimensions
    '''

    valueChanged = QtCore.pyqtSignal(list)

    def __init__(self, name, labels_x, default_value, optional, tooltip, labels_y=None, parent=None):
        super(OptionMultiBox, self).__init__(parent)

        #TODO: separate dtype for iterable of iterables where order of duts is selected

        # different color palettes to visualize disabled widgets
        # disabled
        self.palette_dis = QtGui.QPalette()
        self.palette_dis.setColor(QtGui.QPalette.Base, QtCore.Qt.gray)
        self.palette_dis.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
        # enabled
        self.palette_en = QtGui.QPalette()
        self.palette_en.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
        self.palette_en.setColor(QtGui.QPalette.Text, QtCore.Qt.black)

        nx = len(labels_x)
        ny = len(labels_y) if labels_y else 1

        # Check default value
        if default_value is None:  # None is only supported for all values
            default_value = [[None] * ny] * nx

        # Option name name with boxes below
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(name)
        if tooltip:
            text.setToolTip(tooltip)
        if optional:  # Values can be unset
            layout_1 = QtWidgets.QHBoxLayout()
            layout_1.addWidget(text)
            layout_1.addStretch(0)
            check_box_opt = QtWidgets.QCheckBox()
            layout_1.addWidget(check_box_opt)
            layout.addLayout(layout_1)
        else:
            layout.addWidget(text)

        # Layout for boxes
        layout_iter = QtWidgets.QGridLayout()
        offset = 1 if labels_y else 0

        # Matrix with checkboxes
        self.check_boxes = []

        for i, label in enumerate(labels_x):
            layout_iter.addWidget(QtWidgets.QLabel(label), 0, i + offset, alignment=QtCore.Qt.AlignCenter)
            tmp = []
            for j in range(ny):
                if labels_y and not i:
                    layout_iter.addWidget(QtWidgets.QLabel('  ' + labels_y[j]), j + 1, 0)
                check_box = QtWidgets.QCheckBox()
                check_box.stateChanged.connect(self._emit_value)
                if name == 'Align duts':
                    check_box.clicked.connect(lambda: self._evaluate_state())
                if ny == 1:
                    self.check_boxes.append(check_box)
                else:
                    tmp.append(check_box)
                layout_iter.addWidget(check_box, j + 1, i + offset, alignment=QtCore.Qt.AlignCenter)
            if ny != 1:
                self.check_boxes.append(tmp)
        layout.addLayout(layout_iter)

        # set default values
        for i in range(len(default_value)):
            if isinstance(default_value[i], collections.Iterable):
                for j in range(len(default_value[i])):
                    if default_value[i][j] is not None:
                        self.check_boxes[i][default_value[i][j]].setChecked(True)
            else:
                if default_value[i] is not None:
                    self.check_boxes[default_value[i]].setChecked(True)

        if name == 'Align duts':
            self._evaluate_state(init=True)

        if optional:
            check_box_opt.stateChanged.connect(lambda v: self._set_readonly(v == 0))
            if name == 'Align duts':
                check_box_opt.stateChanged.connect(lambda: self._evaluate_state(init=True))
            self._set_readonly()

    def _evaluate_state(self, init=False):

        if init:
            empty_rows = []
            for i in range(len(self.check_boxes)):
                tmp = []
                for j in range(len(self.check_boxes[i])):
                    if not self.check_boxes[j][i].isChecked():
                        self.check_boxes[j][i].setDisabled(True)
                        self.check_boxes[j][i].setPalette(self.palette_dis)
                    else:
                        tmp.append(j)

                if not tmp:
                    empty_rows.append(i)

            for row in empty_rows:
                self.check_boxes[0][row].setChecked(True)

        else:
            for i in range(len(self.check_boxes)):
                    if self.sender() in self.check_boxes[i]:
                        sender_col = i
                        sender_row = self.check_boxes[i].index(self.sender())

            for j in range(len(self.check_boxes)):
                if j != sender_col:
                    if self.check_boxes[j][sender_row].isEnabled():
                        self.check_boxes[j][sender_row].setChecked(False)
                        self.check_boxes[j][sender_row].setDisabled(True)
                        self.check_boxes[j][sender_row].setPalette(self.palette_dis)
                    else:
                        self.check_boxes[j][sender_row].setDisabled(False)
                        self.check_boxes[j][sender_row].setPalette(self.palette_en)

    def _get_values(self):

        values = []
        for i in range(len(self.check_boxes)):
            tmp = []
            if isinstance(self.check_boxes[i], collections.Iterable):
                for j in range(len(self.check_boxes[i])):
                    if self.check_boxes[i][j].isChecked():
                        tmp.append(j)
            else:
                if self.check_boxes[i].isChecked():
                    values.append(i)
                    continue
            if tmp:
                values.append(tmp)

        if not values:
            values = [None]

        return values

    def _set_readonly(self, value=True):
        if value:
            for cb in self.check_boxes:
                if isinstance(cb, collections.Iterable):
                    for cb_1 in cb:
                        cb_1.setPalette(self.palette_dis)
                        cb_1.setDisabled(True)
                else:
                    cb.setPalette(self.palette_dis)
                    cb.setDisabled(True)
            self._emit_value()
        else:
            for cb in self.check_boxes:
                if isinstance(cb, collections.Iterable):
                    for cb_1 in cb:
                        cb_1.setPalette(self.palette_en)
                        cb_1.setDisabled(False)
                else:
                    cb.setPalette(self.palette_en)
                    cb.setDisabled(False)
            self._emit_value()

    def _emit_value(self):
        values = self._get_values()
        self.valueChanged.emit(values)
