import logging
import matplotlib
import inspect

from PyQt5 import QtWidgets, QtCore

matplotlib.use('Qt5Agg')  # Make sure that we are using QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class AnalysisPlotter(QtWidgets.QWidget):
    """
    Implements generic plotting area widget. Takes one or multiple plotting functions and their input files
    and displays figures from their return values. Supports single and multiple figures as return values.
    Also supports plotting from multiple functions at once and input of predefined figures
    """

    def __init__(self, input_file, plot_func, figures=None, parent=None, **kwargs):

        super(AnalysisPlotter, self).__init__(parent)

        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input arguments
        self.input_file = input_file
        self.plot_func = plot_func
        self.figures = figures
        self.kwargs = kwargs

        # Bool whether to plot from multiple functions at once
        multi_plot = False

        # Multiple plot_functions with respective input_data; dicts of plotting functions and input files
        # must have same keys. If figures are given, they must be given as a dict with a key that is in the
        # plot_functions keys. If kwargs are given for the plotting functions, keyword must be in plot_functions keys.
        # Value must be dict with actual kwarg for plot function. Full example for multi plotting WITH kwargs
        # for each plotting function would look like:
        #
        #    self.input_file={'event': input_file_event, 'correlation': input_file_correlations}
        #    self.plot_func={'event': plot_events, 'correlation': plot_correlations}
        #    self.kwargs={'event': {'event_range': 40}, 'correlation':{'pixel_size':(250,50), 'dut_names':'Tel_0'}}
        #
        # which is equivalent to:
        #
        #    AnalysisPlotter(self.input_files, self.plot_func, event={'event_range': 40}, correlation={'pixel_size':(250,50), 'dut_names':'Tel_0'})

        if isinstance(self.input_file, dict) and isinstance(self.plot_func, dict):
            if self.input_file.keys() != self.plot_func.keys():
                msg = 'Different sets of keys! Can not assign input data to respective plotting function!'
                logging.error(msg=msg)
                raise KeyError
            else:
                if self.kwargs:
                    for key in self.kwargs.keys():
                        if key not in self.plot_func.keys():
                            msg = 'Can not assign keyword %s with argument %s to any plotting function.' \
                                  ' Keyword must be in keys of plotting function dictionary: %s.' \
                                  % (key, str(self.kwargs[key]), ''.join(str(self.plot_func.keys())))
                            raise KeyError(msg)

                multi_plot = True

        # Whether to plot a single or multiple functions
        if not multi_plot:

            # Check whether kwargs are are args in plot_func
            if self.kwargs:
                self.check_kwargs(self.plot_func, self.kwargs)

            if self.figures is None:
                self.plot()
            else:
                self.plot(figures=self.figures)
        else:

            if self.kwargs:

                # Check whether kwargs are are args in plot_func
                for key in self.kwargs.keys():
                    self.check_kwargs(self.plot_func[key], self.kwargs[key])

            self.multi_plot()

    def check_kwargs(self, plot_func, kwargs):

        # Get plot_func's args
        args = inspect.getargspec(plot_func)[0]

        for kw in kwargs.keys():
            if kw not in args:
                msg = 'Plotting function %s got unexpected argument %s' % (plot_func.__name__, kw)
                raise TypeError(msg)
            else:
                pass

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
            fig = self.plot_func(self.input_file, gui=True, **self.kwargs)
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

                    label_count.setText('%d of %d' % (index + 1, plot_widget.count()))

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
                if key in self.kwargs.keys():
                    fig = self.plot_func[key](self.input_file[key], gui=True, **self.kwargs[key])
                else:
                    fig = self.plot_func[key](self.input_file[key], gui=True)

            self.plot(external_widget=dummy_widget, figures=fig)

            tabs.addTab(dummy_widget, str(key).capitalize())

        self.main_layout.addWidget(tabs)