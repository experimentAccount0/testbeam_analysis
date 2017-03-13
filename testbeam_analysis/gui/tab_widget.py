''' Defines all analysis tabs

    Each tab is for one analysis function and has function
    gui options and plotting outputs
'''

from testbeam_analysis.gui.analysis_widget import AnalysisWidget
from testbeam_analysis.hit_analysis import generate_pixel_mask, cluster_hits
from testbeam_analysis.dut_alignment import correlate_cluster, merge_cluster_data, prealignment, apply_alignment, alignment
from testbeam_analysis.track_analysis import find_tracks, fit_tracks


class NoisyPixelsTab(AnalysisWidget):
    ''' Implements the noisy pixel analysis gui'''

    def __init__(self, parent, setup, options):
        super(NoisyPixelsTab, self).__init__(
            parent, setup, options, input_file=None)

        self.add_function(func=generate_pixel_mask)


class ClusterPixelsTab(AnalysisWidget):
    ''' Implements the pixel clustering gui'''

    def __init__(self, parent, setup, options):
        super(ClusterPixelsTab, self).__init__(
            parent, setup, options, input_file=None)

        self.add_function(func=cluster_hits)


class PrealignmentTab(AnalysisWidget):
    ''' Implements the prealignment gui'''

    def __init__(self, parent, setup, options):
        super(PrealignmentTab, self).__init__(
            parent, setup, options, input_file=None)

        self.add_function(func=correlate_cluster)
        self.add_function(func=prealignment)


class TrackFindingTab(AnalysisWidget):
    ''' Implements the track finding gui'''

    def __init__(self, parent, setup, options):
        super(TrackFindingTab, self).__init__(
            parent, setup, options, input_file=None)

        self.add_function(func=find_tracks)


class AlignmentTab(AnalysisWidget):
    ''' Implements the alignment gui'''

    def __init__(self, parent, setup, options):
        super(AlignmentTab, self).__init__(
            parent, setup, options, input_file=None)

        self.add_function(func=alignment)
        self.add_option(option='TEST', func=alignment, dtype='bool',
                        name='HAMMA', optional=False, default_value=True)


class TrackFittingTab(AnalysisWidget):
    ''' Implements the track fitting gui'''

    def __init__(self, parent, setup, options):
        super(TrackFittingTab, self).__init__(
            parent, setup, options, input_file=None)

        self.add_function(func=fit_tracks)
