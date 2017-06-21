import yaml
import traceback

from multiprocessing import Pool
from PyQt5 import QtCore


class AnalysisThread(QtCore.QThread):
    """
    Implements a class which allows to perform analysis / start vitables on an
    extra thread to keep the GUI responsive during analysis / vitables
    """

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
        """ 
        Runs the function func with given argument args. If funcs_args is not None, it contains
        functions and corresponding arguments which are looped over and run. If errors or exceptions
        occur, a signal sends the exception to main thread. Most recent traceback wil be dumped in yaml file.
        """

        try:

            if self.funcs_args is not None:
                pool = Pool()
                for func, kwargs in self.funcs_args:
                    pool.apply_async(self.main_func(func, kwargs))
                pool.close()
                pool.join()

            else:
                self.main_func(self.args)

        except Exception as e:

            # Save the latest traceback on sub thread to file
            trc_bck = traceback.format_exc()
            with open('traceback.yaml', 'w') as f_write:
                yaml.dump(trc_bck, f_write, default_flow_style=False)

            self.exceptionSignal.emit(e)


class AnalysisWorker(QtCore.QObject):
    """
    Implements a worker class which allows the worker to perform analysis / start vitables
    while being moved to an extra thread to keep the GUI responsive during analysis / vitables
    """

    finished = QtCore.pyqtSignal()
    exceptionSignal = QtCore.pyqtSignal(Exception)

    def __init__(self, func, args=None, funcs_args=None):
        QtCore.QObject.__init__(self)

        # Main function which will be executed on this thread
        self.main_func = func
        # Arguments of main function
        self.args = args
        # Functions and arguments to perform analysis function;
        # if not None, main function is then AnalysisWidget.call_funcs()
        self.funcs_args = funcs_args

    def work(self):
        """ 
        Runs the function func with given argument args. If funcs_args is not None, it contains
        functions and corresponding arguments which are looped over and run. If errors or exceptions
        occur, a signal sends the exception to main thread. Most recent traceback wil be dumped in yaml file.
        """

        try:

            if self.funcs_args is not None:
                pool = Pool()
                for func, kwargs in self.funcs_args:
                    pool.apply_async(self.main_func(func, kwargs))
                pool.close()
                pool.join()

            else:
                self.main_func(self.args)

            self.finished.emit()

        except Exception as e:

            # Save the latest traceback on sub thread to file
            trc_bck = traceback.format_exc()
            with open('traceback.yaml', 'w') as f_write:
                yaml.dump(trc_bck, f_write, default_flow_style=False)

            self.exceptionSignal.emit(e)