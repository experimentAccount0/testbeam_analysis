"""
Implements a worker object on which analysis can be done. The worker is then moved to a separate QThread
via the QObject.moveToThread() method.
"""

import traceback

from multiprocessing import Pool
from PyQt5 import QtCore


class AnalysisWorker(QtCore.QObject):
    """
    Implements a worker class which allows the worker to perform analysis / start vitables
    while being moved to an extra thread to keep the GUI responsive during analysis / vitables
    """

    finished = QtCore.pyqtSignal()
    exceptionSignal = QtCore.pyqtSignal(Exception, str)
    statusSignal = QtCore.pyqtSignal(int)

    def __init__(self, func, args=None, funcs_args=None):
        super(AnalysisWorker, self).__init__(self)

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

        # Counter to emit status of analysis
        index_status = 0

        try:

            # Do analysis functions
            if self.funcs_args is not None:

                pool = Pool()
                for func, kwargs in self.funcs_args:

                    # If kwargs is list, do func for each element in kwargs; used for parallel analysis
                    if isinstance(kwargs, list):

                        for k in kwargs:
                            pool.apply_async(self.main_func(func, k))

                            # Increase index and emit signal
                            index_status += 1
                            self.statusSignal.emit(index_status)

                    # Each func has unique kwargs; used for analysis
                    else:
                        pool.apply_async(self.main_func(func, kwargs))

                        # Increase index and emit signal
                        index_status += 1
                        self.statusSignal.emit(index_status)

                pool.close()
                pool.join()

            # Do some arbitrary function
            else:
                self.main_func(self.args)

            self.finished.emit()

        except Exception as e:

            # Format traceback and send
            trc_bck = traceback.format_exc()

            # Emit exception signal
            self.exceptionSignal.emit(e, trc_bck)
