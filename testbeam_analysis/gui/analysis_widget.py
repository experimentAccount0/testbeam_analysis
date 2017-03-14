import os
import inspect
import logging
import math
from collections import OrderedDict
from numpydoc.docscrape import FunctionDoc
from PyQt5 import QtWidgets, QtCore, QtGui

from testbeam_analysis.gui import option_widget


def get_default_args(func):
    """
    Returns a dictionary of arg_name:default_values for the input function
    """
    args, _, _, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))


def get_parameter_doc(func, dtype=False):
    ''' Returns a dictionary of paramerter:pardoc for the input function

        Pardoc is either the parameter description (dtype=False) or the
        data type (dtype=False)
    '''
    doc = FunctionDoc(func)
    pars = {}
    for par, datatype, descr in doc['Parameters']:
        if not dtype:
            pars[par] = '\n'.join(descr)
        else:
            pars[par] = datatype
    return pars


class AnalysisWidget(QtWidgets.QWidget):
    ''' Implements a generic analysis gui.

        There are two seperated widget areas. One the left one for plotting
        and on the right for function parameter options.
        There are 3 kind of options:
          - needed ones on top
          - optional options that can be deactivated below
          - fixed option that cannot be changed
        Below this is a button to call the underlying function with given
        keyword arguments from the options.

        Introprospection is used to determine function argument types and
        documentation from the function implementation automatically.
    '''

    def __init__(self, parent, setup, options, input_file, multi=False):
        super(AnalysisWidget, self).__init__(parent)
        self.setup = setup
        self.options = options
        self.input_file = input_file
        self.option_widgets = {}
        self.multi = multi
        self._setup()
        # Holds functions with kwargs
        self.calls = OrderedDict()

    def _setup(self):
        # Plot area
        left_widget = QtWidgets.QWidget(parent=self)
        self.plt = QtWidgets.QHBoxLayout()
        left_widget.setLayout(self.plt)
        # Options
        self.opt_needed = QtWidgets.QVBoxLayout()
        self.opt_optional = QtWidgets.QVBoxLayout()
        self.opt_fixed = QtWidgets.QVBoxLayout()
        # Option area
        layout_options = QtWidgets.QVBoxLayout()
        self.label_option = QtWidgets.QLabel('Options')
        layout_options.addWidget(self.label_option)
        layout_options.addLayout(self.opt_needed)
        layout_options.addLayout(self.opt_optional)
        layout_options.addLayout(self.opt_fixed)
        layout_options.addStretch(0)
        # Proceed button
        button_ok = QtWidgets.QPushButton('OK')
        button_ok.clicked.connect(self._call_funcs)
        layout_options.addWidget(button_ok)
        right_widget = QtWidgets.QWidget(parent=self)
        right_widget.setLayout(layout_options)
        # Split plot and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 2.5)
        widget_splitter.setChildrenCollapsible(False)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def _option_exists(self, option):
        ''' Check if option is already defined
        '''
        for call in self.calls.values():
            for kwarg in call:
                if option == kwarg:
                    return True
        return False

    def add_options_auto(self, func):
        ''' Inspect a function to create options for kwargs
        '''
        for name in get_default_args(func):
            # Only add as function parameter if the info is not
            # given in setup/option data structures
            if name in self.setup:
                if not self._option_exists(option=name):
                    self.add_option(option=name, default_value=self.setup[name],
                                    func=func, fixed=True)
                else:
                    self.calls[func][name] = self.setup[name]
            elif name in self.options:
                if not self._option_exists(option=name):
                    self.add_option(option=name, default_value=self.options[name],
                                    func=func, fixed=True)
                else:
                    self.calls[func][name] = self.options[name]
            else:
                self.add_option(func=func, option=name)

    def add_option(self, option, func, dtype=None, name=None, optional=None, default_value=None, fixed=False, tooltip=None):
        ''' Add an option to the gui to set function arguments

            option: str
                Function argument name
            func: function
                Function to be used for the option
            dtype: str
                Type string to select proper input method, if None determined from default parameter type
            name: str
                Name shown in gui
            optional: bool
                Show as optional option, If optional is not defined all parameters with default value
                None are set as optional. The common behavior is that None deactivates a parameter
            default_value : object
                Default value for option
            fixed : boolean
                Fix option value  default value
        '''
        # Check if option exists already
        if option in self.calls[func]:
            self._delete_option(option=option, func=func)

        # Get name from argument name
        if not name:
            name = option.replace("_", " ").capitalize()

        # Get default argument value
        if default_value is None:
            default_value = get_default_args(func)[option]

        # Get parameter description from numpy style docstring
        if not tooltip:
            try:
                tooltip = get_parameter_doc(func)[option]
            except KeyError:  # No parameter docu available
                logging.warning(
                    'Parameter %s in function %s not documented', option, func.__name__)
                tooltip = None

        # Get parameter dtype from numpy style docstring
        if not dtype:
            try:
                dtype = get_parameter_doc(func, dtype=True)[option]
            except KeyError:  # No parameter docu available
                pass

        # Get dtype from default arg
        if not dtype:
            if default_value is not None:
                dtype = str(type(default_value).__name__)
            else:
                raise RuntimeError(
                    'Cannot deduce data type for %s in function %s, because no default parameter exists', option, func.__name__)

        # Get optional argument from default function argument
        if optional is None and default_value is None:
            optional = True

        if not fixed:  # Option value can be changed
            # Create widget according to data type
            if 'str' in dtype:
                widget = option_widget.OptionText(
                    name, default_value, optional, tooltip, parent=self)
            elif 'int' in dtype:
                widget = option_widget.OptionSlider(
                    name, default_value, optional, tooltip, parent=self)
            elif 'float' in dtype:
                widget = option_widget.OptionSlider(
                    name, default_value, optional, tooltip, parent=self)
            elif 'bool' in dtype:
                widget = option_widget.OptionBool(
                    name, default_value, optional, tooltip, parent=self)
            elif ('scalar' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                  ('iterable' in dtype and 'iterable of iterable' not in dtype)):
                widget = option_widget.OptionMultiSlider(
                    name=name, labels=self.setup[
                        'dut_names'], default_value=default_value,
                    optional=optional, tooltip=tooltip, parent=self)
            else:
                logging.warning(
                    'Cannot create option %s for dtype "%s" for function %s', option, dtype, func.__name__)
                return
    #             raise NotImplementedError('Cannot use type %s', dtype)

            self._set_argument(
                func, option, default_value if not optional else None)
            self.option_widgets[option] = widget
            self.option_widgets[option].valueChanged.connect(
                lambda value: self._set_argument(func, option, value))

            if optional:
                self.opt_optional.addWidget(self.option_widgets[option])
            else:
                self.opt_needed.addWidget(self.option_widgets[option])
        else:  # Fixed value
            if default_value is None:
                raise RuntimeError(
                    'Cannot create fixed option without default value')
            text = QtWidgets.QLabel()
            # Fixed options cannot be changed --> grey color
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkGray)
            text.setPalette(palette)
            text.setToolTip(tooltip)
            text.setText(name + ': ' + str(default_value))
            self.opt_fixed.addWidget(text)
            self.calls[func][option] = default_value

    def _delete_option(self, option, func):
        ''' Delete existing option. Needed if option is set manually.
        '''
        # Delete option widget
        self.option_widgets[option].close()
        del self.option_widgets[option]
        # Update widgets
        self.opt_optional.update()
        self.opt_needed.update()
        # Delete kwarg
        del self.calls[func][option]

    def add_function(self, func):
        ''' Add an analysis function'''
        self.calls[func] = {}
        # Add tooltip from function docstring
        doc = FunctionDoc(func)
        label_option = self.label_option.toolTip()
        self.label_option.setToolTip(label_option +
                                     '\n'.join(doc['Summary']))
        # Add function options to gui
        self.add_options_auto(func)

    def _set_argument(self, func, name, value):
        # Workaround for https://www.riverbankcomputing.com/pipermail/pyqt/2016-June/037662.html
        # Cannot transmit None for signals with string (likely also float)
        if type(value) == str and 'None' in value:
            value = None
        if type(value) == float and math.isnan(value):
            value = None
        if type(value) == list and None in value:
            value = None
        self.calls[func][name] = value

    def _call_func(self, func, kwargs):
        ''' Call an analysis function with given kwargs

            Setup info and generic options are added if needed.
        '''

        # Set missing kwargs from setting data structures
        args = inspect.getargspec(func)[0]
        for arg in args:
            if arg not in self.calls[func]:
                if arg in self.setup:
                    kwargs[arg] = self.setup[arg]
                elif arg in self.options or 'file' in arg:
                    try:
                        if 'input' in arg or 'output' in arg:
                            kwargs[arg] = os.path.join(self.options['working_directory'],
                                                       self.options[arg])
                        else:
                            kwargs[arg] = self.options[arg]
                    except KeyError:
                        logging.error(
                            'File I/O %s not defined in settings', arg)
                else:
                    raise RuntimeError('Function argument %s not defined', arg)
        print func.__name__, kwargs
        func(**kwargs)

    def _call_funcs(self):
        ''' Call all functions in a row
        '''

        for func, kwargs in self.calls.iteritems():
            print func.__name__, kwargs
            #self._call_func(func, kwargs)
