import sys
import logging

from PyQt5 import QtCore


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


class LogBuffer(object):
    """
    An object that listens to whatever is put in queue
    """

    def __init__(self, queue):
        self.queue = queue

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, log):
        self.queue.put(log)


class LogReceiver(QtCore.QObject):
    """
    A receiver that sits on its own thread and waits for input from a queue, then emits signal with input
    """

    sendLog = QtCore.pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):

        QtCore.QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    @QtCore.pyqtSlot()
    def start_receiver(self):
        while True:
            log = self.queue.get()
            self.sendLog.emit(log)