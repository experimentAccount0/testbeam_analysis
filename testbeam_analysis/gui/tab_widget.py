""" Defines all analysis tabs

    Each tab is for one analysis function and has function
    gui options and plotting outputs
"""

from PyQt5 import QtCore, QtWidgets, QtGui
from testbeam_analysis.gui.analysis_widgets import AnalysisWidget, ParallelAnalysisWidget
from testbeam_analysis.hit_analysis import generate_pixel_mask, cluster_hits
from testbeam_analysis.dut_alignment import correlate_cluster, prealignment, merge_cluster_data, apply_alignment, alignment
from testbeam_analysis.track_analysis import find_tracks, fit_tracks
from testbeam_analysis.result_analysis import calculate_efficiency, calculate_residuals

# Plot related import
from testbeam_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_size, plot_correlations, plot_events


class NoisyPixelsTab(ParallelAnalysisWidget):
    """ Implements the noisy pixel analysis gui"""

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(NoisyPixelsTab, self).__init__(parent, setup, options, tab_list)

        # Make options and setup class variables
        self.options = options
        self.setup = setup

        # Make variables for input of noisy pixel function
        self.output_files = [options['output_path'] + '/' + dut + options['noisy_suffix'] for dut in setup['dut_names']]
        self.input_files = options['input_files']
        self.n_pixels = setup['n_pixels']
        self.duts = setup['dut_names']

        self.add_parallel_function(func=generate_pixel_mask)

        self.add_parallel_option(option='input_hits_file',
                                 default_value=self.input_files,
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='output_mask_file',
                                 default_value=self.output_files,
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='n_pixel',
                                 default_value=self.n_pixels,
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='dut_name',
                                 default_value=self.duts,
                                 func=generate_pixel_mask,
                                 fixed=False)

        # Add checkbox to each tab to enable skipping noisy pixel removal individually
        self.check_boxes = {}
        for dut in self.tw.keys():
            cb = QtWidgets.QCheckBox('Skip noisy pixel removal for %s' % dut)
            self.check_boxes[dut] = cb
            self.tw[dut].layout_options.addWidget(self.check_boxes[dut])

        # Disconnect ok button and reconnect
        self.btn_ok.disconnect()
        self.btn_ok.clicked.connect(lambda: self.check_skip_noisy())

    def check_skip_noisy(self):
        """
        Checks whether or not the noisy pixel masking is skipped for specific DUTs. Changes the respective
        input parameters for the analysis widget.
        """

        # Clear noisy pixel input variables
        self.input_files = []
        self.output_files = []
        self.n_pixels = []
        self.duts = []

        # Loop over tabs and check the state of checkbox
        for dut in self.setup['dut_names']:

            # Mask noisy pixels for files with un-checked boxes
            if not self.check_boxes[dut].isChecked():

                self.input_files.append(self.options['input_files'][self.setup['dut_names'].index(dut)])
                self.output_files.append(self.options['output_path'] + '/' + dut + self.options['noisy_suffix'])
                self.duts.append(dut)
                self.n_pixels.append(self.setup['n_pixels'][self.setup['dut_names'].index(dut)])

            # Do nothing
            else:
                pass

        # Do plotting and connect vitables if masking is done for at least one DUT
        if self.input_files:

            for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                      lambda: self._connect_vitables(files=self.output_files),
                      lambda: self.plot(input_files=self.output_files,
                                        plot_func=plot_masked_pixels,
                                        dut_names=self.duts)]:
                self.parallelAnalysisDone.connect(x)

            # Start masking
            self._call_parallel_funcs()

        # If all DUTs are skipped, disable container and enable next tab
        else:

            for key in self.tw.keys():
                self.tw[key].container.setDisabled(True)

            self.btn_ok.setDisabled(True)

            self.parallelAnalysisDone.connect(lambda _tab_list: self.proceedAnalysis.emit(_tab_list))
            self.parallelAnalysisDone.emit(self.tab_list)


class ClusterPixelsTab(ParallelAnalysisWidget):
    ''' Implements the pixel clustering gui'''

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(ClusterPixelsTab, self).__init__(parent, setup, options, tab_list)

        output_files = [options['output_path'] + '/' + dut + options['cluster_suffix'] for dut in setup['dut_names']]

        self.add_parallel_function(func=cluster_hits)

        self.add_parallel_option(option='input_hits_file',
                                 default_value=options['input_files'],
                                 func=cluster_hits,
                                 fixed=True)

        self.add_parallel_option(option='input_noisy_pixel_mask_file',
                                 default_value=[options['output_path'] + '/' + dut + options['noisy_suffix'] for dut in setup['dut_names']],
                                 func=cluster_hits,
                                 fixed=True)

        self.add_parallel_option(option='output_cluster_file',
                                 default_value=output_files,
                                 func=cluster_hits,
                                 fixed=True)

        self.add_parallel_option(option='dut_name',
                                 default_value=setup['dut_names'],
                                 func=cluster_hits,
                                 fixed=False)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_files),
                  lambda: self.plot(input_files=output_files,
                                    plot_func=plot_cluster_size)]:
            self.parallelAnalysisDone.connect(x)


class PrealignmentTab(AnalysisWidget):
    ''' Implements the prealignment gui. Prealignment uses
        4 functions of test beam analysis:
        - correlate cluster
        - fit correlations (prealignment)
        - merge cluster data of duts
        - apply prealignment
    '''

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(PrealignmentTab, self).__init__(parent, setup, options, tab_list)

        output_files = {'correlation': options['output_path'] + '/Correlation.h5',
                        'alignment': options['output_path'] + '/Alignment.h5',
                        'merged': options['output_path'] + '/Merged.h5',
                        'tracklets': options['output_path'] + '/Tracklets_prealigned.h5'}

        multiple_plotting_data = {'correlation': output_files['correlation']}
        multiple_plotting_func = {'correlation': plot_correlations}

        self.add_function(func=correlate_cluster)
        self.add_function(func=prealignment)
        self.add_function(func=merge_cluster_data)
        self.add_function(func=apply_alignment)

        self.add_option(option='input_cluster_files',
                        default_value=[options['output_path'] + '/' + dut + options['cluster_suffix'] for dut in setup['dut_names']],
                        func=correlate_cluster,
                        fixed=True)

        self.add_option(option='output_correlation_file',
                        default_value=output_files['correlation'],
                        func=correlate_cluster,
                        fixed=True)

        self.add_option(option='input_correlation_file',
                        default_value=output_files['correlation'],
                        func=prealignment,
                        fixed=True)

        self.add_option(option='output_alignment_file',
                        default_value=output_files['alignment'],
                        func=prealignment,
                        fixed=True)

        self.add_option(option='input_cluster_files',
                        default_value=[options['output_path'] + '/' + dut + options['cluster_suffix'] for dut in setup['dut_names']],
                        func=merge_cluster_data,
                        fixed=True)

        self.add_option(option='output_merged_file',
                        default_value=output_files['merged'],
                        func=merge_cluster_data,
                        fixed=True)

        self.add_option(option='input_hit_file',
                        default_value=output_files['merged'],
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=output_files['alignment'],
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='output_hit_file',
                        default_value=output_files['tracklets'],
                        func=apply_alignment,
                        fixed=True)

        # Fix options that should not be changed
        self.add_option(option='use_duts', func=apply_alignment,
                        default_value=[1] * setup['n_duts'], fixed=True)
        self.add_option(option='inverse', func=apply_alignment, fixed=True)
        self.add_option(option='force_prealignment', func=apply_alignment,
                        default_value=True, fixed=True)
        self.add_option(option='no_z', func=apply_alignment, fixed=True)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_files.values()),
                  lambda: self.plot(input_file=multiple_plotting_data,
                                    plot_func=multiple_plotting_func,
                                    correlation={'dut_names': setup['dut_names']})]:  # kwargs for correlation
            self.analysisDone.connect(x)


class TrackFindingTab(AnalysisWidget):
    ''' Implements the track finding gui'''

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(TrackFindingTab, self).__init__(parent, setup, options, tab_list)

        output_file = options['output_path'] + '/TrackCandidates_prealignment.h5'

        self.add_function(func=find_tracks)

        self.add_option(option='input_tracklets_file',
                        default_value=options['output_path'] + '/Tracklets_prealigned.h5',
                        func=find_tracks,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=options['output_path'] + '/Alignment.h5',
                        func=find_tracks,
                        fixed=True)

        self.add_option(option='output_track_candidates_file',
                        default_value=output_file,
                        func=find_tracks,
                        fixed=True)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_file),
                  lambda: self.plot(input_file=output_file, plot_func=plot_events, dut=0, event_range=(0, 10))]:
            self.analysisDone.connect(x)


class AlignmentTab(AnalysisWidget):
    ''' Implements the alignment gui'''

    proceedAnalysis = QtCore.pyqtSignal(list)
    skipAlignment = QtCore.pyqtSignal()

    def __init__(self, parent, setup, options, tab_list):
        super(AlignmentTab, self).__init__(parent, setup, options, tab_list)

        if isinstance(tab_list, list):
            self.tl = tab_list
        else:
            self.tl = [tab_list]

        output_file = options['output_path'] + '/Tracklets.h5'

        self.add_function(func=alignment)
        self.add_function(func=apply_alignment)

        self.add_option(option='input_track_candidates_file',
                        default_value=options['output_path'] + '/TrackCandidates_prealignment.h5',
                        func=alignment,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=options['output_path'] + '/Alignment.h5',
                        func=alignment,
                        fixed=True)

        self.add_option(option='initial_translation',
                        default_value=False,
                        func=alignment,
                        fixed=True)

        self.add_option(option='initial_rotation',
                        default_value=setup['rotations'],
                        func=alignment,
                        fixed=True)

        self.add_option(option='input_hit_file',
                        default_value=options['output_path'] + '/Merged.h5',
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=options['output_path'] + '/Alignment.h5',
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='output_hit_file',
                        default_value=output_file,
                        func=apply_alignment,
                        fixed=True)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_file),
                  lambda: self.btn_skip.deleteLater()]:
            self.analysisDone.connect(x)

        self.btn_skip = QtWidgets.QPushButton('Skip')
        self.btn_skip.setToolTip('Skip alignment and use pre-alignment for further analysis')
        self.btn_skip.clicked.connect(lambda: self._skip_alignment())
        self.layout_options.addWidget(self.btn_skip)
        self.btn_ok.clicked.connect(lambda: self.btn_skip.setDisabled(True))

        # When global settings are updated, recreate state of alignment tab
        if options['skip_alignment']:
            self._skip_alignment(ask=False)

    def _skip_alignment(self, ask=True):

        if ask:
            msg = 'Do you want to skip alignment and use pre-alignment for further analysis?'
            reply = QtWidgets.QMessageBox.question(self, 'Skip alignment', msg, QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.Cancel)
        else:
            reply = QtWidgets.QMessageBox.Yes

        if reply == QtWidgets.QMessageBox.Yes:

            self.btn_skip.setText('Alignment skipped')
            self.btn_ok.deleteLater()
            self.container.setDisabled(True)

            if ask:
                self.skipAlignment.emit()
                self.proceedAnalysis.emit(self.tl)
        else:
            pass


class TrackFittingTab(AnalysisWidget):
    ''' Implements the track fitting gui'''

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(TrackFittingTab, self).__init__(parent, setup, options, tab_list)

        if options['skip_alignment']:
            input_tracks = options['output_path'] + '/TrackCandidates_prealignment.h5'
            output_file = options['output_path'] + '/Tracks_prealigned.h5'
        else:
            output_file = options['output_path'] + '/Tracks_aligned.h5'

            self.add_function(func=find_tracks)

            self.add_option(option='input_tracklets_file',
                            default_value=options['output_path'] + '/Tracklets.h5',  # from alignment
                            func=find_tracks,
                            fixed=True)

            self.add_option(option='input_alignment_file',
                            default_value=options['output_path'] + '/Alignment.h5',
                            func=find_tracks,
                            fixed=True)

            self.add_option(option='output_track_candidates_file',
                            default_value=options['output_path'] + '/TrackCandidates.h5',
                            func=find_tracks,
                            fixed=True)

            input_tracks = options['output_path'] + '/TrackCandidates.h5'

        self.add_function(func=fit_tracks)

        self.add_option(option='input_track_candidates_file',
                        default_value=input_tracks,
                        func=fit_tracks,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=options['output_path'] + '/Alignment.h5',
                        func=fit_tracks,
                        fixed=True)

        self.add_option(option='output_tracks_file',
                        default_value=output_file,
                        func=fit_tracks,
                        fixed=True)

        # Set and fix options
        self.add_option(option='fit_duts', func=fit_tracks,
                        default_value=range(setup['n_duts']), optional=True)
        self.add_option(option='force_prealignment', func=fit_tracks,
                        default_value=options['skip_alignment'], fixed=True)
        self.add_option(option='exclude_dut_hit', func=fit_tracks,
                        default_value=False, fixed=True)
        self.add_option(option='use_correlated', func=fit_tracks,
                        default_value=False, fixed=True)
        self.add_option(option='min_track_distance', func=fit_tracks,
                        default_value=[200] * setup['n_duts'], optional=False)

        # Check whether scatter planes in setup
        if setup['scatter_planes']['sct_names']:
            self.add_option(option='add_scattering_plane',
                            default_value=setup['scatter_planes'],
                            func=fit_tracks,
                            fixed=True)
        else:
            self.add_option(option='add_scattering_plane',
                            default_value=False,
                            func=fit_tracks,
                            fixed=True)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_file)]:
            self.analysisDone.connect(x)


class ResidualTab(AnalysisWidget):
    ''' Implements the result analysis gui'''

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(ResidualTab, self).__init__(parent, setup, options, tab_list)

        if options['skip_alignment']:
            input_tracks = options['output_path'] + '/Tracks_prealigned.h5'
        else:
            input_tracks = options['output_path'] + '/Tracks_aligned.h5'

        self.add_function(func=calculate_residuals)

        output_file = options['output_path'] + '/Residuals.h5'

        self.add_option(option='input_tracks_file',
                        default_value=input_tracks,
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=options['output_path'] + '/Alignment.h5',
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='output_residuals_file',
                        default_value=output_file,
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='force_prealignment',
                        default_value=options['skip_alignment'],
                        func=calculate_residuals,
                        fixed=True)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_file)]:
            self.analysisDone.connect(x)


class EfficiencyTab(AnalysisWidget):
    """
    Implements the efficiency results tab
    """

    proceedAnalysis = QtCore.pyqtSignal(list)

    def __init__(self, parent, setup, options, tab_list):
        super(EfficiencyTab, self).__init__(parent, setup, options, tab_list)

        if options['skip_alignment']:
            input_tracks = options['output_path'] + '/Tracks_prealigned.h5'
        else:
            input_tracks = options['output_path'] + '/Tracks_aligned.h5'

        self.add_function(func=calculate_efficiency)

        output_file = options['output_path'] + '/Efficiency.h5'

        self.add_option(option='input_tracks_file',
                        default_value=input_tracks,
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=options['output_path'] + '/Alignment.h5',
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='output_residuals_file',
                        default_value=output_file,
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='bin_size',
                        default_value=setup['pixel_size'],
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='sensor_size',
                        default_value=[(setup['pixel_size'][i][0] * setup['n_pixels'][i][0],
                                        setup['pixel_size'][i][1] * setup['n_pixels'][i][1])
                                       for i in range(len(setup['dut_names']))],
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='force_prealignment',
                        default_value=options['skip_alignment'],
                        func=calculate_efficiency,
                        fixed=True)

        for x in [lambda _tab_list: self.proceedAnalysis.emit(_tab_list),
                  lambda: self._connect_vitables(files=output_file)]:
            self.analysisDone.connect(x)