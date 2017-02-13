import os
import inspect
import logging
import math
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
        self.keywords = {}
        self.option_widgets = {}
        self.multi = multi
        self._setup()

    def _setup(self):
        # Plot area
        left_widget = QtWidgets.QWidget()
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
        button_ok.clicked.connect(self._call_func)
        layout_options.addWidget(button_ok)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(layout_options)
        # Split plot and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 2.5)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def add_options_auto(self):
        for name in get_default_args(self.func):
            # Only add as function parameter if the info is not
            # given in setup/option data structures
            if name in self.setup:
                self.add_fixed_option(option=name, value=self.setup[name])
            elif name in self.options:
                self.add_fixed_option(option=name, value=self.options[name])
            else:
                self.add_option(option=name)

    def add_option(self, option, dtype=None, name=None, optional=None):
        ''' Add an option to the gui to set function arguments

            option: str
                Function argument name
            dtype: str
                Type string to select proper input method, if None determined from default parameter type
            name: str
                Name shown in gui
            optional: bool
                Show as optional option, If optional is not defined all parameters with default value
                None are set as optional. The common behavior is that None deactivates a parameter
        '''

        # Get name from argument name
        if not name:
            name = option.replace("_", " ").capitalize()

        # Get default argument value
        default_value = get_default_args(self.func)[option]

        # Get parameter description from numpy style docstring
        try:
            tooltip = get_parameter_doc(self.func)[option]
        except KeyError:  # No parameter docu available
            tooltip = None

        # Get parameter dtype from numpy style docstring
        if not dtype:
            try:
                dtype = get_parameter_doc(self.func, dtype=True)[option]
            except KeyError:  # No parameter docu available
                pass

        # Get dtype from default arg
        if not dtype:
            if default_value:
                dtype = str(type(default_value).__name__)
            else:
                raise RuntimeError(
                    'Cannot deduce data type for %s, because no default parameter exists', option)

        # Get optional argument from default function argument
        if optional is None and default_value is None:
            optional = True
        else:
            optional = False

        # Create widget according to data type
        if 'str' in dtype:
            widget = option_widget.OptionText(
                name, default_value, optional, tooltip)
        elif 'int' in dtype:
            widget = option_widget.OptionSlider(
                name, default_value, optional, tooltip)
        elif 'float' in dtype:
            widget = option_widget.OptionSlider(
                name, default_value, optional, tooltip)
        elif 'bool' in dtype:
            widget = option_widget.OptionBool(
                name, default_value, optional, tooltip)
        else:
            logging.warning(
                'Cannot create option %s for dtype %s', option, dtype)
            return
#             raise NotImplementedError('Cannot use type %s', dtype)

        self._set_argument(option, default_value)

        self.option_widgets[option] = widget

        self.option_widgets[option].valueChanged.connect(
            lambda value: self._set_argument(option, value))

        if optional:
            self.opt_optional.addWidget(self.option_widgets[option])
        else:
            self.opt_needed.addWidget(self.option_widgets[option])

    def set_option(self, option, dtype=None, name=None, optional=None):
        ''' Change existing option. Needed if automation fails.

            For parameters see add_option()
        '''
        # Delete option widget
        self.option_widgets[option].close()
        del self.option_widgets[option]
        del self.keywords[option]
        # Update widgets
        self.opt_optional.update()
        self.opt_needed.update()
        # Set new widget
        self.add_option(option=option, dtype=dtype,
                        name=name, optional=optional)

    def add_fixed_option(self, option, value, name=None):
        # Get name from argument name
        if not name:
            name = option.replace("_", " ").capitalize()
        text = QtWidgets.QLabel()
        # Fixed options cannot be changed --> grey color
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkGray)
        text.setPalette(palette)
        try:
            tooltip = get_parameter_doc(self.func)[option]
            text.setToolTip(tooltip)
        except KeyError:  # No parameter docu available
            pass
        text.setText(name + ': ' + str(value))
        self.opt_fixed.addWidget(text)

    def set_function(self, func):
        ''' Set the analysis function'''
        self.func = func
        # Set tooltip from function docstring
        doc = FunctionDoc(self.func)
        self.label_option.setToolTip('\n'.join(doc['Summary']))
        # Add function options to gui
        self.add_options_auto()

    def _set_argument(self, name, value):
        # Workaround for https://www.riverbankcomputing.com/pipermail/pyqt/2016-June/037662.html
        # Cannot transmit None for signals with string (likely also float)
        if type(value) == str and 'None' in value:
            value = None
        if type(value) == float and math.isnan(value):
            value = None
        self.keywords[name] = value

    def _call_func(self):
        ''' Call the analysis function with options from gui

            Setup info and generic options are added if needed.
        '''

        # Set missing kwargs from setting data structures
        args = inspect.getargspec(self.func)[0]
        for arg in args:
            if arg not in self.keywords:
                if arg in self.setup:
                    self.keywords[arg] = self.setup[arg]
                elif arg in self.options:
                    if 'input' in arg:
                        if 'file' in arg:
                            self.keywords[arg] = os.path.join(
                                self.options['working_directory'], self.options[arg])
                        else:
                            self.keywords[arg] = self.options[arg]
                # Never plot
                elif arg == 'plot':
                    self.keywords['plot'] = False
                else:
                    raise RuntimeError('Function argument %s not defined', arg)

        print self.keywords
        self.func(**self.keywords)
