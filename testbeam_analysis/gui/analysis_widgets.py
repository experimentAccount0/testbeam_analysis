import os
import sys
import inspect
import logging
import math

from multiprocessing import Pool
from subprocess import call, CalledProcessError
from collections import OrderedDict
from numpydoc.docscrape import FunctionDoc
from PyQt5 import QtWidgets, QtCore, QtGui

from testbeam_analysis.gui import option_widget

import matplotlib
matplotlib.use('Qt5Agg')  # Make sure that we are using QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def get_default_args(func):
    """
    Returns a dictionary of arg_name:default_values for the input function
    """
    args, _, _, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))


def get_parameter_doc(func, dtype=False):
    """ 
    Returns a dictionary of paramerter:pardoc for the input function
    Pardoc is either the parameter description (dtype=False) or the data type (dtype=False)
    """
    doc = FunctionDoc(func)
    pars = {}
    for par, datatype, descr in doc['Parameters']:
        if not dtype:
            pars[par] = '\n'.join(descr)
        else:
            pars[par] = datatype
    return pars


class AnalysisWidget(QtWidgets.QWidget):
    """
    Implements a generic analysis gui.

    There are two separated widget areas. One the left one for plotting
    and on the right for function parameter options.
    There are 3 kind of options:
      - needed ones on top
      - optional options that can be deactivated below
      - fixed option that cannot be changed
    Below this is a button to call the underlying function with given
    keyword arguments from the options.

    Introprospection is used to determine function argument types and
    documentation from the function implementation automatically.
    """

    # Signal emitted after all funcs are called
    analysisDone = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list=None):
        super(AnalysisWidget, self).__init__(parent)
        self.setup = setup
        self.options = options
        self.option_widgets = {}
        self.splitter_size = [parent.width()/2, parent.width()/2]
        self._setup()
        # Provide additional thread to do analysis on
        self.analysis_thread = None
        # Provide additional thread for vitables
        self.vitables_thread = None
        # Holds functions with kwargs
        self.calls = OrderedDict()
        # List of tabs which will be enabled after analysis
        if isinstance(tab_list, list):
            self.tab_list = tab_list
        else:
            self.tab_list = [tab_list]

    def _setup(self):
        # Plot area
        self.left_widget = QtWidgets.QWidget()
        self.plt = QtWidgets.QVBoxLayout()
        self.left_widget.setLayout(self.plt)
        # Options
        self.opt_needed = QtWidgets.QVBoxLayout()
        self.opt_optional = QtWidgets.QVBoxLayout()
        self.opt_fixed = QtWidgets.QVBoxLayout()
        # Option area
        self.layout_options = QtWidgets.QVBoxLayout()
        self.label_option = QtWidgets.QLabel('Options')
        self.layout_options.addWidget(self.label_option)
        self.layout_options.addLayout(self.opt_needed)
        self.layout_options.addLayout(self.opt_optional)
        self.layout_options.addLayout(self.opt_fixed)
        self.layout_options.addStretch(0)

        # Proceed button
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(self._call_funcs)

        # Container widget to disable all but ok button after perfoming analysis
        self.container = QtWidgets.QWidget()
        self.container.setLayout(self.layout_options)

        # Right widget
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add container and ok button to right widget
        self.right_widget.layout().addWidget(self.container)
        self.right_widget.layout().addWidget(self.btn_ok)

        # Make right widget scroll able
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setBackgroundRole(QtGui.QPalette.Light)
        scroll.setWidget(self.right_widget)

        # Split plot and option area
        self.widget_splitter = QtWidgets.QSplitter(parent=self)
        self.widget_splitter.addWidget(self.left_widget)
        self.widget_splitter.addWidget(scroll)
        self.widget_splitter.setSizes(self.splitter_size)
#        self.widget_splitter.setStretchFactor(0, 10)
#        self.widget_splitter.setStretchFactor(1, 2.5)
        self.widget_splitter.setChildrenCollapsible(False)
        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(self.widget_splitter)
        self.setLayout(layout_widget)

    def _option_exists(self, option):
        """
        Check if option is already defined
        """
        for call in self.calls.values():
            for kwarg in call:
                if option == kwarg:
                    return True
        return False

    def add_options_auto(self, func):
        """
        Inspect a function to create options for kwargs
        """

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
        """
        Add an option to the gui to set function arguments

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
        """

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
            try:
                widget = self._select_widget(dtype, name, default_value,
                                             optional, tooltip)
            except NotImplementedError:
                logging.warning('Cannot create option %s for dtype "%s" for function %s',
                                option, dtype, func.__name__)
                return

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
            text.setWordWrap(True)
            # Fixed options cannot be changed --> grey color
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkGray)
            text.setPalette(palette)
            text.setToolTip(tooltip)
            # Handle displaying list entries of default_value
            if isinstance(default_value, list):
                # Needed to get width of text
                metrics = QtGui.QFontMetrics(self.font())
                # If width of default_value as str is greater than widget, make new line for each entry
                if metrics.width(str(default_value)) > self.right_widget.width():
                    d_v = ['\n' + str(v) for v in default_value]
                    t = name + ':' + ''.join(d_v) + '\n'
                # If not, write list in one line
                else:
                    t = (name + ':\n' + str(default_value) + '\n')
                text.setText(t)
            else:
                text.setText(name + ':\n' + str(default_value) + '\n')
            self.opt_fixed.addWidget(text)
            self.calls[func][option] = default_value

    def _select_widget(self, dtype, name, default_value, optional, tooltip):
        # Create widget according to data type
        if ('scalar' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                'int' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                ('iterable' in dtype and 'iterable of iterable' not in dtype)):
            widget = option_widget.OptionMultiSlider(
                name=name, labels=self.setup['dut_names'],
                default_value=default_value,
                optional=optional, tooltip=tooltip, parent=self)
        elif 'iterable of iterable' in dtype:
            widget = option_widget.OptionMultiBox(
                name=name, labels_x=self.setup['dut_names'],
                default_value=default_value,
                optional=optional, tooltip=tooltip, labels_y=self.setup['dut_names'], parent=self)
        elif 'str' in dtype:
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
        else:
            raise NotImplementedError('Cannot use type %s', dtype)

        return widget

    def _delete_option(self, option, func):
        """
        Delete existing option. Needed if option is set manually.
        """

        # Delete option widget
        self.option_widgets[option].close()
        del self.option_widgets[option]
        # Update widgets
        self.opt_optional.update()
        self.opt_needed.update()
        # Delete kwarg
        del self.calls[func][option]

    def add_function(self, func):
        """
        Add an analysis function
        """

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
        """
        Call an analysis function with given kwargs
        Setup info and generic options are added if needed.
        """

        # Set missing kwargs from setting data structures
        args = inspect.getargspec(func)[0]
        for arg in args:
            if arg not in self.calls[func]:
                if arg in self.setup:
                    kwargs[arg] = self.setup[arg]
                elif arg in self.options or 'file' in arg:
                    try:
                        if 'input' in arg or 'output' in arg:
                            kwargs[arg] = os.path.join(self.options['output_path'],  # self.options['working_directory']
                                                       self.options[arg])
                        else:
                            kwargs[arg] = self.options[arg]
                    except KeyError:
                        logging.error(
                            'File I/O %s not defined in settings', arg)
                else:
                    raise RuntimeError('Function argument %s not defined', arg)
        # print(func.__name__, kwargs)
        func(**kwargs)

    def _call_funcs(self):
        """ 
        Call all functions in a row
        """

        try:
            self.btn_ok.setDisabled(True)
        except RuntimeError:
            pass

        # Should work but analysis_thread somehow returns "finished" signal before all task are done

        #self.worker = AnalysisWorker(func=self._call_func, funcs_args=self.calls.iteritems())
        #self.worker.moveToThread(self.analysis_thread)
        #self.worker.finished.connect(lambda: self.analysisDone.emit(self.tab_list))
        #self.worker.run_call_funcs()

        #self.analysis_thread = AnalysisThread(func=self._call_func, funcs_args=self.calls.iteritems())
        #self.analysis_thread.finished.connect(lambda: self.analysis_thread.quit())
        #self.analysis_thread.finished.connect(lambda: self.analysisDone.emit(self.tab_list))
        #self.analysis_thread.start()

        # Comment rest of this function when trying multithreading

        pool = Pool()
        for func, kwargs in self.calls.iteritems():
            # print(func.__name__, kwargs)
            pool.apply_async(self._call_func(func, kwargs))
        pool.close()
        pool.join()

        # Emit signal to indicate end of analysis
        if self.tab_list is not None:
            self.analysisDone.emit(self.tab_list)
            self.container.setDisabled(True)

    def _connect_vitables(self, files):

        self.btn_ok.setDisabled(False)
        self.btn_ok.setText('Open output file(s) via ViTables')
        self.btn_ok.clicked.disconnect()
        self.btn_ok.clicked.connect(lambda: self._call_vitables(files=files))

    def _call_vitables(self, files):

        if isinstance(files, list):
            vitables_paths = ['vitables']
            for f in files:
                vitables_paths.append(str(f))
        else:
            vitables_paths = ['vitables', str(files)]

        self.vitables_thread = AnalysisThread(func=call, args=vitables_paths)
        self.vitables_thread.exceptionSignal.connect(lambda exception: self.handle_exceptions(exception=exception,
                                                                                              cause='vitables'))
        self.vitables_thread.exceptionSignal.connect(lambda: self.vitables_thread.quit())
        self.vitables_thread.start()

    def plot(self, input_file, plot_func, dut_names=None, figures=None):

        plot = AnalysisPlotter(input_file=input_file, plot_func=plot_func, dut_names=dut_names, figures=figures, parent=self.left_widget)
        self.plt.addWidget(plot)

    def handle_exceptions(self, exception, cause=None):

        if cause is not None:

            if cause == 'vitables':

                if exception is CalledProcessError:
                    msg = 'An error occurred during executing ViTables'
                elif type(exception) is OSError:
                    msg = 'ViTables not found. Try installing ViTables'
                    self.btn_ok.setToolTip('Try installing or re-installing ViTables')
                    self.btn_ok.setText('ViTables not found')
                    self.btn_ok.setDisabled(True)
                else:
                    raise exception

            elif cause == 'analysis':

                if exception is CalledProcessError:
                    msg = 'An error occurred during analysis'
                else:
                    raise exception

            logging.error(msg)

        else:
            raise exception


class ParallelAnalysisWidget(QtWidgets.QWidget):
    """
    AnalysisWidget for functions that need to run for every input data file.
    Creates UI with one tab widget per respective input file
    """

    parallelAnalysisDone = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list=None):

        super(ParallelAnalysisWidget, self).__init__(parent)

        # Make main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Add sub-layout and ok button and progressbar
        self.sub_layout = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(lambda: self._call_parallel_funcs())
        self.p_bar = QtWidgets.QProgressBar()
        self.p_bar.setVisible(False)

        # Set alignment in sub-layout
        self.sub_layout.addWidget(self.p_bar)
        self.sub_layout.addWidget(self.btn_ok)
        self.sub_layout.setAlignment(self.p_bar, QtCore.Qt.AlignLeading)
        self.sub_layout.setAlignment(self.btn_ok, QtCore.Qt.AlignTrailing)

        # Tab related widgets
        self.tabs = QtWidgets.QTabWidget()
        self.tw = {}

        # Add to main layout
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addLayout(self.sub_layout)

        # Initialize options and setup
        self.setup = setup
        self.options = options

        # Additional thread for vitables
        self.vitables_thread = None

        # List of tabs which will be enabled after analysis
        if isinstance(tab_list, list):
            self.tab_list = tab_list
        else:
            self.tab_list = [tab_list]

        self._init_tabs()
        self.connect_tabs()

    def _init_tabs(self):

        # Clear widgets
        self.tabs.clear()
        self.tw = {}

        for i in range(self.setup['n_duts']):

            tmp_setup = {}
            tmp_options = {}

            for s_key in self.setup.keys():

                if isinstance(self.setup[s_key], list) or isinstance(self.setup[s_key], tuple):
                    if isinstance(self.setup[s_key][i], str):
                        tmp_setup[s_key] = [self.setup[s_key][i]]  # FIXME: Does not work properly without list
                    else:
                        tmp_setup[s_key] = self.setup[s_key][i]
                elif isinstance(self.setup[s_key], int) or isinstance(self.setup[s_key], str):
                    tmp_setup[s_key] = self.setup[s_key]

            for o_key in self.options.keys():

                if isinstance(self.options[o_key], list) or isinstance(self.options[o_key], tuple):
                    if isinstance(self.options[o_key][i], str):
                        tmp_options[o_key] = [self.options[o_key][i]]  # FIXME: Does not work properly without list
                    else:
                        tmp_options[o_key] = self.options[o_key][i]
                elif isinstance(self.options[o_key], int) or isinstance(self.options[o_key], str):
                    tmp_options[o_key] = self.options[o_key]

            widget = AnalysisWidget(parent=self.tabs, setup=tmp_setup, options=tmp_options, tab_list=self.tab_list)
            widget.btn_ok.deleteLater()

            self.tw[self.setup['dut_names'][i]] = widget
            self.tabs.addTab(self.tw[self.setup['dut_names'][i]], self.setup['dut_names'][i])

    def connect_tabs(self):

        self.tabs.currentChanged.connect(lambda tab: self.handle_sub_layout(tab=tab))

        for tab_name in self.tw.keys():
            self.tw[tab_name].widget_splitter.splitterMoved.connect(
                lambda: self.handle_sub_layout(tab=self.tabs.currentIndex()))

    def resizeEvent(self, QResizeEvent):
        self.handle_sub_layout(tab=self.tabs.currentIndex())

    def showEvent(self, QShowEvent):
        self.handle_sub_layout(tab=self.tabs.currentIndex())

    def handle_sub_layout(self, tab):

        offset = 10
        sub_widths = self.tw[self.tabs.tabText(tab)].widget_splitter.sizes()

        self.p_bar.setFixedWidth(sub_widths[0] + offset)
        self.btn_ok.setFixedWidth(sub_widths[1] + offset)

    def add_parallel_function(self, func):
        for i in range(self.setup['n_duts']):
            self.tw[self.setup['dut_names'][i]].add_function(func=func)

    def add_parallel_option(self, option, default_value, func, name=None, dtype=None, optional=None, fixed=False, tooltip=None):

        for i in range(self.setup['n_duts']):
            self.tw[self.setup['dut_names'][i]].add_option(option=option, func=func, dtype=dtype, name=name,
                                                           optional=optional, default_value=default_value[i],
                                                           fixed=fixed, tooltip=tooltip)

    def _call_parallel_funcs(self):

        self.btn_ok.setDisabled(True)

        self.p_bar.setRange(0, len(self.tw.keys()))
        self.p_bar.setVisible(True)

        for i, tab in enumerate(self.tw.keys()):
            self.p_bar.setValue(i+1)
            self.tw[tab]._call_funcs()

        if self.tab_list is not None:
            self.parallelAnalysisDone.emit(self.tab_list)

    def _connect_vitables(self, files):

        self.btn_ok.setDisabled(False)
        self.btn_ok.setText('Open output file(s) via ViTables')
        self.btn_ok.clicked.disconnect()
        self.btn_ok.clicked.connect(lambda: self._call_vitables(files=files))

    def _call_vitables(self, files):

        if isinstance(files, list):
            vitables_paths = ['vitables']
            for f in files:
                vitables_paths.append(str(f))
        else:
            vitables_paths = ['vitables', str(files)]

        self.vitables_thread = AnalysisThread(func=call, args=vitables_paths)
        self.vitables_thread.exceptionSignal.connect(lambda exception: self.handle_exceptions(exception=exception,
                                                                                              cause='vitables'))
        self.vitables_thread.exceptionSignal.connect(lambda: self.vitables_thread.quit())
        self.vitables_thread.start()

    def plot(self, input_files, plot_func, dut_names=None, figures=None):

        if dut_names is not None:
            if isinstance(dut_names, list):
                names = dut_names
            else:
                names = [dut_names]
        else:
            names = list(self.tw.keys())
            names.reverse()

        for dut in names:
            plot = AnalysisPlotter(input_file=input_files[names.index(dut)], plot_func=plot_func, dut_names=dut, figures=figures)
            self.tw[dut].plt.addWidget(plot)

    def handle_exceptions(self, exception, cause=None):

        if cause is not None:

            if cause == 'vitables':

                if exception is CalledProcessError:
                    msg = 'An error occurred during executing ViTables'
                elif type(exception) is OSError:
                    msg = 'ViTables not found. Try installing ViTables'
                    self.btn_ok.setToolTip('Try installing or re-installing ViTables')
                    self.btn_ok.setText('ViTables not found')
                    self.btn_ok.setDisabled(True)
                else:
                    raise exception

            elif cause == 'analysis':

                if exception is CalledProcessError:
                    msg = 'An error occurred during analysis'
                else:
                    raise exception

            logging.error(msg)

        else:
            raise exception


class AnalysisPlotter(QtWidgets.QWidget):
    """
    Implements plotting area widget. Takes one or multiple plotting functions and their input files
    and displays figures from their return values. Supports single and multiple figures as return values.
    Also supports plotting from multiple functions at once and input of predefined figures
    """

    def __init__(self, input_file, plot_func, dut_names=None, figures=None, parent=None):

        super(AnalysisPlotter, self).__init__(parent)

        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input arguments
        self.input_file = input_file
        self.plot_func = plot_func
        self.dut_names = dut_names
        self.figures = figures

        # Bool whether to plot from multiple functions at once
        multi_plot = False

        # Multiple plot_functions with respective input_data; function and input file must have same key
        # If figures are given, they must be given as a dict with a key that is in the plot function keys
        if isinstance(self.input_file, dict) and isinstance(self.plot_func, dict):
            if self.input_file.keys() != self.plot_func.keys():
                msg = 'Different sets of keys! Can not assign input data to respective plotting function!'
                logging.error(msg=msg)
                raise KeyError
            else:
                multi_plot = True

        # Whether to plot a single or multiple functions
        if not multi_plot:
            if self.figures is None:
                self.plot()
            else:
                self.plot(figures=self.figures)
        else:
            self.multi_plot()

    def plot(self, external_widget=None, figures=None):
        """
        Function for plotting one or multiple plots from a single plot_func.
        If the function returns multiple plots, respective widgets for navigation
        through plots are created.
        
        :param external_widget: None or QWidget; if None figs are plotted on self (single fig) or an internal
                                plot_widget. If QWidget figs are plotted on this widget (must have layout)
                                
        :param figures: Figure() or list of Figures(); if None figures come from the return values of self.plot_func.
                        If figures are given, just plot them onto widget.
        """

        # Get plots
        if figures is None:
            # Different kwarg for some plotting funcs: dut_name vs dut_names
            try:
                fig = self.plot_func(self.input_file, dut_names=self.dut_names, gui=True)
            except TypeError:
                fig = self.plot_func(self.input_file, dut_name=self.dut_names, gui=True)
        else:
            fig = figures

        # Make list of figures if not already
        if isinstance(fig, list):
            fig_list = fig
        else:
            fig_list = [fig]

        # Check for multiple plots and init plot widget
        if len(fig_list) > 1:
            plot_widget = QtWidgets.QStackedWidget()
        else:
            # Plots will be on self or external_widget
            plot_widget = None

        # Create a dummy widget and add a figure canvas and a toolbar for each plot
        for f in fig_list:
            dummy_widget = QtWidgets.QWidget()
            dummy_layout = QtWidgets.QVBoxLayout()
            dummy_widget.setLayout(dummy_layout)
            f.set_facecolor('0.99')
            canvas = FigureCanvas(f)
            canvas.setParent(self)
            toolbar = NavigationToolbar(canvas, self)
            dummy_layout.addWidget(toolbar)
            dummy_layout.addWidget(canvas)

            # Handle plot_widget and amount of figs
            if isinstance(plot_widget, QtWidgets.QStackedWidget):  # Multiple figs
                plot_widget.addWidget(dummy_widget)
            else:  # Single fig
                if external_widget is None:  # Plot on self
                    self.main_layout.addWidget(dummy_widget)
                else:  # Plot on external_widget
                    external_widget.layout().addWidget(dummy_widget)

        # If more than one fig make navigation widgets and add everything to respective widgets
        if isinstance(plot_widget, QtWidgets.QStackedWidget):

            # Add plot widget to external widget or self
            if external_widget is None:
                self.main_layout.addWidget(plot_widget)
            else:
                external_widget.layout().addWidget(plot_widget)

            # Create buttons to navigate through different plots
            layout_btn = QtWidgets.QHBoxLayout()
            btn_forward = QtWidgets.QPushButton()
            btn_back = QtWidgets.QPushButton()
            icon_forward = btn_forward.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward)
            icon_back = btn_back.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack)
            btn_forward.setIcon(icon_forward)
            btn_back.setIcon(icon_back)
            btn_forward.setIconSize(QtCore.QSize(40, 40))
            btn_back.setIconSize(QtCore.QSize(40, 40))
            label_count = QtWidgets.QLabel('1 of %d' % plot_widget.count())
            # Connect buttons
            btn_forward.clicked.connect(lambda: navigate(val=1))
            btn_back.clicked.connect(lambda: navigate(val=-1))
            # Add buttons to layout
            layout_btn.addStretch()
            layout_btn.addWidget(btn_back)
            layout_btn.addSpacing(20)
            layout_btn.addWidget(label_count)
            layout_btn.addSpacing(20)
            layout_btn.addWidget(btn_forward)
            layout_btn.addStretch()

            # Disable back button when at first plot
            if plot_widget.currentIndex() == 0:
                btn_back.setDisabled(True)

            # Add all to main or external layout
            if external_widget is None:
                self.main_layout.addLayout(layout_btn)
            else:
                external_widget.layout().addLayout(layout_btn)

            # button slot to change plots
            def navigate(val):

                if 0 <= (plot_widget.currentIndex() + val) <= plot_widget.count():
                    index = plot_widget.currentIndex() + val
                    plot_widget.setCurrentIndex(index)

                    if index == plot_widget.count() - 1:
                        btn_back.setDisabled(False)
                        btn_forward.setDisabled(True)
                    elif index == 0:
                        btn_back.setDisabled(True)
                        btn_forward.setDisabled(False)
                    else:
                        btn_forward.setDisabled(False)
                        btn_back.setDisabled(False)

                    label_count.setText('%d of %d' % (index+1, plot_widget.count()))

                else:
                    pass

    def multi_plot(self):
        """
        Function that allows plotting from multiple plot functions at once.
        Creates a tab widget and one tab for every plot function. Uses self.plot() to plot
        """

        if self.figures is not None:

            if isinstance(self.figures, dict):
                pass
            else:
                msg = 'Input figures must be in dictionary! Can not assign figure(s) to respective plotting function!'
                logging.error(msg=msg)
                raise KeyError

        tabs = QtWidgets.QTabWidget()

        for key in self.input_file.keys():

            dummy_widget = QtWidgets.QWidget()
            dummy_widget.setLayout(QtWidgets.QVBoxLayout())

            if self.figures is not None and key in self.figures.keys():

                fig = self.figures[key]

            else:

                # Different kwarg for some plotting funcs: dut_name vs dut_names
                try:
                    fig = self.plot_func[key](self.input_file[key], dut_names=self.dut_names, gui=True)
                except TypeError:
                    fig = self.plot_func[key](self.input_file[key], dut_name=self.dut_names, gui=True)

            self.plot(external_widget=dummy_widget, figures=fig)

            tabs.addTab(dummy_widget, str(key).capitalize())

        self.main_layout.addWidget(tabs)


class AnalysisStream(QtCore.QObject):
    """
    Class to handle the stdout stream which is used to do thread safe logging
    since QtWidgets are not thread safe and therefore one can not directly log to GUIs
    widgets when performing analysis on different thread than main thread
    """

    _stdout = None
    _stderr = None
    messageWritten = QtCore.pyqtSignal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(unicode(msg))

    @staticmethod
    def stdout():
        if not AnalysisStream._stdout:
            AnalysisStream._stdout = AnalysisStream()
            sys.stdout = AnalysisStream._stdout
        return AnalysisStream._stdout

    @staticmethod
    def stderr():
        if not AnalysisStream._stderr:
            AnalysisStream._stderr = AnalysisStream()
            sys.stderr = AnalysisStream._stderr
        return AnalysisStream._stderr


class AnalysisLogger(logging.Handler):
    """
    Implements a logging handler which allows redirecting log thread-safe
    """

    def __init__(self, parent):

        super(AnalysisLogger, self).__init__()

    def emit(self, record):
        msg = self.format(record)
        if msg:
            AnalysisStream.stdout().write(msg)


class AnalysisThread(QtCore.QThread):
    """
    Implements a class which allows to perform analysis / start vitables on an
    extra thread to keep the GUI responsive during analysis / vitables
    """

    analysisProgress = QtCore.pyqtSignal(int)
    exceptionSignal = QtCore.pyqtSignal(Exception)

    def __init__(self, func, args=None, funcs_args=None, parent=None):

        super(AnalysisThread, self).__init__(parent)

        # Main function which will be executed on this thread
        self.main_func = func
        # Arguments of main function
        self.args = args
        # Functions and arguments to perform analysis function;
        # if not None, main function is then AnalysisWidget.call_funcs()
        self.funcs_args = funcs_args

    def run(self):

        try:

            if self.funcs_args is not None:

                pool = Pool()
                for func, kwargs in self.funcs_args:
                    self.analysisProgress.emit(1)
                    pool.apply_async(self.main_func(func, kwargs))
                pool.close()
                pool.join()

            else:
                self.analysisProgress.emit(0)
                self.main_func(self.args)

        except Exception as e:
            self.exceptionSignal.emit(e)


class AnalysisWorker(QtCore.QObject):

    finished = QtCore.pyqtSignal()

    def __init__(self, func, args=None, funcs_args=None, parent=None):

        super(AnalysisWorker, self).__init__(parent)

        self.main_func = func
        self.funcs_args = funcs_args
        self.args = args

    @QtCore.pyqtSlot()
    def run_call_funcs(self):

        pool = Pool()
        for func, kwargs in self.funcs_args:
            # self.main_func(func, kwargs)
            pool.apply_async(self.main_func(func, kwargs))
        pool.close()
        pool.join()

        self.finished.emit()

    @QtCore.pyqtSlot()
    def run_func(self):
        self.main_func(self.args)

        self.finished.emit()
