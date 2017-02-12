''' Defines all analysis tabs

    Each tab is for one analysis function and has function
    gui options and plotting outputs
'''

from testbeam_analysis.gui.analysis_widget import AnalysisWidget
from testbeam_analysis.hit_analysis import generate_pixel_mask, cluster_hits
from testbeam_analysis.dut_alignment import correlate_cluster, merge_cluster_data, prealignment, apply_alignment
from testbeam_analysis.track_analysis import find_tracks, fit_tracks


class NoisyPixelsTab(AnalysisWidget):
    ''' Implements the noisy pixel analysis gui'''

    def __init__(self, parent, setup, options):
        super(NoisyPixelsTab, self).__init__(
            parent, setup, options, input_file=None)

        self.set_function(func=generate_pixel_mask)


class ClusterPixelsTab(AnalysisWidget):
    ''' Implements the pixel clustering gui'''

    def __init__(self, parent, setup, options):
        super(ClusterPixelsTab, self).__init__(
            parent, setup, options, input_file=None)

        self.set_function(func=cluster_hits)


class CorrelateClusterTab(AnalysisWidget):
    ''' Implements the cluster correlation gui'''

    def __init__(self, parent, setup, options):
        super(CorrelateClusterTab, self).__init__(
            parent, setup, options, input_file=None)

        self.set_function(func=correlate_cluster)
