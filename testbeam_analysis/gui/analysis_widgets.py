import os
import inspect
import logging
import math

from subprocess import call, CalledProcessError
from collections import OrderedDict
from numpydoc.docscrape import FunctionDoc
from PyQt5 import QtWidgets, QtCore, QtGui

from testbeam_analysis.gui import option_widget
from analysis_plotter import AnalysisPlotter
from analysis_threading import AnalysisWorker, AnalysisThread

# Set way analysis is done; Can be 'Worker' or 'Thread'
THREADING_MODE = 'Worker'


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
        self.splitter_size = [parent.width() / 2, parent.width() / 2]
        self._setup()

        # FIXME: QThread + WorkerObject
        if THREADING_MODE == 'Worker':
            self.analysis_thread = QtCore.QThread()  # no parent
            self.analysis_worker = None
            self.vitables_thread = QtCore.QThread()  # no parent
            self.vitables_worker = None

        # FIXME: Subclassing QThread
        elif THREADING_MODE == 'Thread':
            # Provide additional thread to do analysis on
            self.analysis_thread = None
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

        # Proceed button and progressbar
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.p_bar = QtWidgets.QProgressBar()
        self.p_bar.setVisible(False)

        for x in [lambda: self._call_funcs(),
                  lambda: scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum()),
                  lambda: self.container.setDisabled(True)]:
            self.btn_ok.clicked.connect(x)

        # Container widget to disable all but ok button after perfoming analysis
        self.container = QtWidgets.QWidget()
        self.container.setLayout(self.layout_options)

        # Scroll area widget
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add container and ok button to right widget
        self.scroll_widget.layout().addWidget(self.container)
        self.scroll_widget.layout().addWidget(self.btn_ok)

        # Make right widget scroll able
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setBackgroundRole(QtGui.QPalette.Light)
        scroll.setWidget(self.scroll_widget)

        # Make widget to hold scroll area and progressbar
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setLayout(QtWidgets.QVBoxLayout())

        self.right_widget.layout().addWidget(scroll)
        self.right_widget.layout().addWidget(self.p_bar)

        # Split plot and option area
        self.widget_splitter = QtWidgets.QSplitter(parent=self)
        self.widget_splitter.addWidget(self.left_widget)
        self.widget_splitter.addWidget(self.right_widget)
        self.widget_splitter.setSizes(self.splitter_size)
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

    def add_option(self, option, func, dtype=None, name=None, optional=None, default_value=None, fixed=False,
                   tooltip=None):
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
                    'Cannot deduce data type for %s in function %s, because no default parameter exists', option,
                    func.__name__)

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
                if metrics.width(str(default_value)) > self.scroll_widget.width():
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

    def _call_funcs(self, parallel=False):
        """ 
        Call all functions in a row
        """

        try:
            self.btn_ok.setDisabled(True)
            self.p_bar.setVisible(True)
            self.p_bar.setRange(0, 0)
        except RuntimeError:
            pass

        # FIXME: This is the QThread + WorkerObject approach which is supposed to be the correct approach
        if THREADING_MODE == 'Worker':
            # Create worker for vitables and move to thread
            self.analysis_worker = AnalysisWorker(func=self._call_func, funcs_args=self.calls.iteritems())
            self.analysis_worker.moveToThread(self.analysis_thread)

            # Connect workers work method to the start of the thread, quit thread when worker finishes
            self.analysis_worker.finished.connect(self.analysis_thread.quit)
            self.analysis_thread.started.connect(self.analysis_worker.work)

            # Connect exceptions signal
            self.analysis_worker.exceptionSignal.connect(lambda e: self.handle_exceptions(exception=e, cause='analysis'))

            # When this method is not called from the ParallelAnalysisWidget
            if not parallel:
                self.analysis_thread.finished.connect(lambda: self.analysisDone.emit(self.tab_list))
                self.analysis_thread.finished.connect(lambda: self.p_bar.setRange(0, 1))
                self.analysis_thread.finished.connect(lambda: self.p_bar.setValue(1))

            # Start thread
            self.analysis_thread.start()

        # FIXME: This is the approach of subclassing QThread which is controversial
        elif THREADING_MODE == 'Thread':
            self.analysis_thread = AnalysisThread(func=self._call_func, funcs_args=self.calls.iteritems())

            # Connect exceptions signal
            self.analysis_thread.exceptionSignal.connect(lambda e: self.handle_exceptions(exception=e, cause='analysis'))

            if not parallel:
                for x in [lambda: self.analysisDone.emit(self.tab_list),
                          lambda: self.p_bar.setRange(0, 1),
                          lambda: self.p_bar.setValue(1)]:
                    self.analysis_thread.finished.connect(x)

            # Start thread
            self.analysis_thread.start()

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

        # FIXME: This is the QThread + WorkerObject approach which should be the correct approach
        if THREADING_MODE == 'Worker':
            # Create worker for vitables and move to thread
            self.vitables_worker = AnalysisWorker(func=call, args=vitables_paths)
            self.vitables_worker.moveToThread(self.vitables_thread)

            # Connect exceptions signal from worker on different thread to main thread
            self.vitables_worker.exceptionSignal.connect(lambda e: self.handle_exceptions(exception=e, cause='vitables'))
            self.vitables_worker.exceptionSignal.connect(self.vitables_thread.quit)

            # Connect workers work method to the start of the thread, quit thread when worker finishes
            self.vitables_worker.finished.connect(self.vitables_thread.quit)
            self.vitables_thread.started.connect(self.vitables_worker.work)

            # Start thread
            self.vitables_thread.start()

        # FIXME: This is the approach of subclassing QThread which is controversial
        elif THREADING_MODE == 'Thread':
            self.vitables_thread = AnalysisThread(func=call, args=vitables_paths)

            # Connect exceptions signal from worker on different thread to main thread
            self.vitables_thread.exceptionSignal.connect(lambda e: self.handle_exceptions(exception=e, cause='vitables'))
            self.vitables_thread.exceptionSignal.connect(self.vitables_thread.quit)

            # Start thread
            self.vitables_thread.start()

    def plot(self, input_file, plot_func, figures=None, **kwargs):

        plot = AnalysisPlotter(input_file=input_file, plot_func=plot_func, figures=figures, parent=self.left_widget, **kwargs)
        self.plt.addWidget(plot)

    def handle_exceptions(self, exception, cause=None):

        if cause is not None:

            if cause == 'vitables':

                if exception is CalledProcessError:
                    msg = 'An error occurred during executing ViTables'
                elif type(exception) in [OSError, ImportError]:
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

        # Flag to check if all analysis threads are finsihed
        self.threads_finished = False

        # Additional thread for vitables
        if THREADING_MODE == 'Worker':
            self.vitables_thread = QtCore.QThread()  # no parent
            self.vitables_worker = None

        elif THREADING_MODE == 'Thread':
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
            widget.p_bar.deleteLater()

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

    def add_parallel_option(self, option, default_value, func, name=None, dtype=None, optional=None, fixed=False,
                            tooltip=None):

        for i in range(self.setup['n_duts']):
            self.tw[self.setup['dut_names'][i]].add_option(option=option, func=func, dtype=dtype, name=name,
                                                           optional=optional, default_value=default_value[i],
                                                           fixed=fixed, tooltip=tooltip)

    def _call_parallel_funcs(self):

        self.btn_ok.setDisabled(True)

        self.p_bar.setRange(0, 0)
        self.p_bar.setVisible(True)

        for tab in self.tw.keys():
            self.tw[tab].container.setDisabled(True)
            self.tw[tab]._call_funcs(parallel=True)
            self.tw[tab].analysis_thread.finished.connect(self._check_parallel_analysis_status)

    def _check_parallel_analysis_status(self):

        if self.p_bar.value() == -1:
            self.p_bar.setRange(0, len(self.tw.keys()))

        analysis_status = []
        for tab in self.tw.keys():
            status = self.tw[tab].analysis_thread.isRunning()
            analysis_status.append(status)

        self.p_bar.setValue(analysis_status.count(False))

        if True not in analysis_status:
            # Due to timing, all analysis threads can be finished while some finished signals are still in queue
            if not self.threads_finished:
                self.parallelAnalysisDone.emit(self.tab_list)
                self.threads_finished = True

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

        # FIXME: This is the QThread + WorkerObject approach which should be the correct approach
        if THREADING_MODE == 'Worker':
            # Create worker for vitables and move to thread
            self.vitables_worker = AnalysisWorker(func=call, args=vitables_paths)
            self.vitables_worker.moveToThread(self.vitables_thread)

            # Connect exceptions signal from worker on different thread to main thread
            self.vitables_worker.exceptionSignal.connect(lambda e: self.handle_exceptions(exception=e, cause='vitables'))
            self.vitables_worker.exceptionSignal.connect(self.vitables_thread.quit)

            # Connect workers work method to the start of the thread, quit thread when worker finishes
            self.vitables_worker.finished.connect(self.vitables_thread.quit)
            self.vitables_thread.started.connect(self.vitables_worker.work)

            # Start thread
            self.vitables_thread.start()

        # FIXME: This is the approach of subclassing QThread which is controversial
        elif THREADING_MODE == 'Thread':
            self.vitables_thread = AnalysisThread(func=call, args=vitables_paths)

            # Connect exceptions signal from worker on different thread to main thread
            self.vitables_thread.exceptionSignal.connect(lambda e: self.handle_exceptions(exception=e, cause='vitables'))
            self.vitables_thread.exceptionSignal.connect(self.vitables_thread.quit)

            # Start thread
            self.vitables_thread.start()

    def plot(self, input_files, plot_func, **kwargs):

        names = list(self.tw.keys())
        names.reverse()

        for dut in names:
            plot = AnalysisPlotter(input_file=input_files[names.index(dut)], plot_func=plot_func, dut_name=dut, **kwargs)
            self.tw[dut].plt.addWidget(plot)

    def handle_exceptions(self, exception, cause=None):

        if cause is not None:

            if cause == 'vitables':

                if exception is CalledProcessError:
                    msg = 'An error occurred during executing ViTables'
                elif type(exception) in [OSError, ImportError]:
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
