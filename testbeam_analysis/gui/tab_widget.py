''' Defines all analysis tabs

    Each tab is for one analysis function and has function
    gui options and plotting outputs
'''

from PyQt5 import QtCore
from testbeam_analysis.gui.analysis_widget import AnalysisWidget, ParallelAnalysisWidget
from testbeam_analysis.hit_analysis import generate_pixel_mask, cluster_hits
from testbeam_analysis.dut_alignment import correlate_cluster, prealignment, merge_cluster_data, apply_alignment, alignment
from testbeam_analysis.track_analysis import find_tracks, fit_tracks
from testbeam_analysis.result_analysis import calculate_efficiency, calculate_residuals


class NoisyPixelsTab(ParallelAnalysisWidget):
    """ Implements the noisy pixel analysis gui"""

    def __init__(self, parent, setup, options):
        super(NoisyPixelsTab, self).__init__(parent, setup, options)

        self.add_parallel_function(func=generate_pixel_mask)

        self.add_parallel_option(option='input_hits_file',
                                 default_value=options['input_files'],
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='output_mask_file',
                                 default_value=[options['output_path'] + '/' + dut + options['noisy_suffix'] for dut in setup['dut_names']],
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='n_pixel',
                                 default_value=setup['n_pixels'],
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='dut_name',
                                 default_value=setup['dut_names'],
                                 func=generate_pixel_mask,
                                 fixed=False)

#        self.update_options(key='input_hits_file',
#                            value=options['input_files'])
#        self.update_options(key='output_mask_file',
#                            value=[options['output_path'] + '/' + dut + options['noisy_suffix'] for dut in setup['dut_names']])
#
#        self.update_setup(key='n_pixel',
#                          value=setup['n_pixels'])
#        self.update_setup(key='dut_name',
#                          value=setup['dut_names'])
#
#        self.add_parallel_function(func=generate_pixel_mask)


class ClusterPixelsTab(ParallelAnalysisWidget):
    ''' Implements the pixel clustering gui'''

    def __init__(self, parent, setup, options, tab_list):
        super(ClusterPixelsTab, self).__init__(parent, setup, options, tab_list)

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
                                 default_value=[options['output_path'] + '/' + dut + options['cluster_suffix'] for dut in setup['dut_names']],
                                 func=cluster_hits,
                                 fixed=True)

        self.add_parallel_option(option='dut_name',
                                 default_value=setup['dut_names'],
                                 func=cluster_hits,
                                 fixed=False)


class PrealignmentTab(AnalysisWidget):
    ''' Implements the prealignment gui. Prealignment uses
        4 functions of test beam analysis:
        - correlate cluster
        - fit correlations (prealignment)
        - merge cluster data of duts
        - apply prealignment
    '''

    proceedAnalysis = QtCore.pyqtSignal('QString')

    def __init__(self, parent, setup, options):
        super(PrealignmentTab, self).__init__(parent, setup, options)

        self.add_function(func=correlate_cluster)
        self.add_function(func=prealignment)
        self.add_function(func=merge_cluster_data)
        self.add_function(func=apply_alignment)

        # Fix options that should not be changed
        self.add_option(option='input_cluster_files', func=correlate_cluster,
                        default_value=[options['output_path'] + name + '_clustered.h5' for name in setup['dut_names']],
                        fixed=True)

#         self.add_option(option='initial_rotation', func=alignment,
#                         default_value=sdf['rotations'], fixed=True)

        self.add_option(option='use_duts', func=apply_alignment,
                        default_value=[1] * setup['n_duts'], fixed=True)
        self.add_option(option='inverse', func=apply_alignment, fixed=True)
        self.add_option(option='force_prealignment', func=apply_alignment,
                        default_value=True, fixed=True)
        self.add_option(option='no_z', func=apply_alignment, fixed=True)


class TrackFindingTab(AnalysisWidget):
    ''' Implements the track finding gui'''

    proceedAnalysis = QtCore.pyqtSignal('QString')

    def __init__(self, parent, setup, options):
        super(TrackFindingTab, self).__init__(parent, setup, options)

        self.add_function(func=find_tracks)


class AlignmentTab(AnalysisWidget):
    ''' Implements the alignment gui'''

    proceedAnalysis = QtCore.pyqtSignal('QString')

    def __init__(self, parent, setup, options):
        super(AlignmentTab, self).__init__(parent, setup, options)

        self.add_function(func=alignment)


class TrackFittingTab(AnalysisWidget):
    ''' Implements the track fitting gui'''

    proceedAnalysis = QtCore.pyqtSignal('QString')

    def __init__(self, parent, setup, options):
        super(TrackFittingTab, self).__init__(parent, setup, options)

        self.add_function(func=fit_tracks)
        # Set and fix options
        self.add_option(option='fit_duts', func=fit_tracks,
                        default_value=[0] * setup['n_duts'], optional=True)
        self.add_option(option='force_prealignment', func=fit_tracks,
                        default_value=False, fixed=True)
        self.add_option(option='exclude_dut_hit', func=fit_tracks,
                        default_value=False, fixed=True)
        self.add_option(option='use_correlated', func=fit_tracks,
                        default_value=False, fixed=True)
        self.add_option(option='min_track_distance', func=fit_tracks,
                        default_value=[200] * setup['n_duts'], optional=False)


class ResultTab(AnalysisWidget):
    ''' Implements the result analysis gui'''

    def __init__(self, parent, setup, options):
        super(ResultTab, self).__init__(parent, setup, options)

        self.add_function(func=calculate_efficiency)
        self.add_function(func=calculate_residuals)
