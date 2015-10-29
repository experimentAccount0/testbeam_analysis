"""This class provides often needed analysis functions, for analysis that is done with python.
"""
import numpy as np
import numexpr as ne
import tables as tb
from math import ceil

from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from testbeam_analysis import analysis_functions
from testbeam_analysis.clusterizer import data_struct


def in1d_events(ar1, ar2):
    """
    Does the same than np.in1d but uses the fact that ar1 and ar2 are sorted and the c++ library. Is therefore much much faster.

    """
    ar1 = np.ascontiguousarray(ar1)  # change memory alignement for c++ library
    ar2 = np.ascontiguousarray(ar2)  # change memory alignement for c++ library
    tmp = np.empty_like(ar1, dtype=np.uint8)  # temporary result array filled by c++ library, bool type is not supported with cython/numpy
    return analysis_functions.get_in1d_sorted(ar1, ar2, tmp)


def get_max_events_in_both_arrays(events_one, events_two):
    """
    Calculates the maximum count of events that exist in both arrays.

    """
    events_one = np.ascontiguousarray(events_one)  # change memory alignement for c++ library
    events_two = np.ascontiguousarray(events_two)  # change memory alignement for c++ library
    event_result = np.empty(shape=(events_one.shape[0] + events_two.shape[0], ), dtype=events_one.dtype)
    count = analysis_functions.get_max_events_in_both_arrays(events_one, events_two, event_result)
    return event_result[:count]


def map_cluster(events, cluster):
    """
    Maps the cluster hits on events. Not existing hits in events have all values set to 0

    """
    cluster = np.ascontiguousarray(cluster)
    events = np.ascontiguousarray(events)
    mapped_cluster = np.zeros((events.shape[0], ), dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
    mapped_cluster = np.ascontiguousarray(mapped_cluster)
    analysis_functions.map_cluster(events, cluster, mapped_cluster)
    return mapped_cluster


def get_events_in_both_arrays(events_one, events_two):
    """
    Calculates the events that exist in both arrays.

    """
    events_one = np.ascontiguousarray(events_one)  # change memory alignement for c++ library
    events_two = np.ascontiguousarray(events_two)  # change memory alignement for c++ library
    event_result = np.empty_like(events_one)
    count = analysis_functions.get_events_in_both_arrays(events_one, events_two, event_result)
    return event_result[:count]


def hist_1d_index(x, shape):
    """
    Fast 1d histogram of 1D indices with C++ inner loop optimization.
    Is more than 2 orders faster than np.histogram().
    The indices are given in coordinates and have to fit into a histogram of the dimensions shape.
    Parameters
    ----------
    x : array like
    shape : tuple
        tuple with x dimensions: (x,)

    Returns
    -------
    np.ndarray with given shape

    """
    if len(shape) != 1:
        raise NotImplementedError('The shape has to describe a 1-d histogram')

    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint32)
    analysis_functions.hist_1d(x, shape[0], result)
    return result


def hist_2d_index(x, y, shape):
    """
    Fast 2d histogram of 2D indices with C++ inner loop optimization.
    Is more than 2 orders faster than np.histogram2d().
    The indices are given in x, y coordinates and have to fit into a histogram of the dimensions shape.
    Parameters
    ----------
    x : array like
    y : array like
    shape : tuple
        tuple with x,y dimensions: (x, y)

    Returns
    -------
    np.ndarray with given shape

    """
    if len(shape) != 2:
        raise NotImplementedError('The shape has to describe a 2-d histogram')

    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    y = np.ascontiguousarray(y.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint32).ravel()  # ravel hist in c-style, 3D --> 1D
    analysis_functions.hist_2d(x, y, shape[0], shape[1], result)
    return np.reshape(result, shape)  # rebuilt 3D hist from 1D hist


def hist_3d_index(x, y, z, shape):
    """
    Fast 3d histogram of 3D indices with C++ inner loop optimization.
    Is more than 2 orders faster than np.histogramdd().
    The indices are given in x, y, z coordinates and have to fit into a histogram of the dimensions shape.
    Parameters
    ----------
    x : array like
    y : array like
    z : array like
    shape : tuple
        tuple with x,y,z dimensions: (x, y, z)

    Returns
    -------
    np.ndarray with given shape

    """
    if len(shape) != 3:
        raise NotImplementedError('The shape has to describe a 3-d histogram')
    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    y = np.ascontiguousarray(y.astype(np.int32))
    z = np.ascontiguousarray(z.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint16).ravel()  # ravel hist in c-style, 3D --> 1D
    analysis_functions.hist_3d(x, y, z, shape[0], shape[1], shape[2], result)
    return np.reshape(result, shape)  # rebuilt 3D hist from 1D hist


def create_2d_pixel_hist(fig, ax, hist2d, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None):
    extent = [0.5, 80.5, 336.5, 0.5]
    if z_max is None:
        if hist2d.all() is np.ma.masked:  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(hist2d.max())
    bounds = np.linspace(start=z_min, stop=z_max, num=255, endpoint=True)
    cmap = cm.get_cmap('jet')
    cmap.set_bad('w')
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(hist2d, interpolation='nearest', aspect="auto", cmap=cmap, norm=norm, extent=extent)
    if title is not None:
        ax.set_title(title)
    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title)
    if y_axis_title is not None:
        ax.set_ylabel(y_axis_title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, boundaries=bounds, cmap=cmap, norm=norm, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), cax=cax)


def get_data_in_event_range(array, event_start=None, event_stop=None, assume_sorted=True):
    '''Selects the data (rows of a table) that occurred in the given event range [event_start, event_stop[

    Parameters
    ----------
    array : numpy.array
    event_start : int, None
    event_stop : int, None
    assume_sorted : bool
        Set to true if the hits are sorted by the event_number. Increases speed.

    Returns
    -------
    numpy.array
        hit array with the hits in the event range.
    '''
    event_number = array['event_number']
    if assume_sorted:
        data_event_start = event_number[0]
        data_event_stop = event_number[-1]
        if (event_start is not None and event_stop is not None) and (data_event_stop < event_start or data_event_start > event_stop or event_start == event_stop):  # special case, no intersection at all
            return array[0:0]

        # get min/max indices with values that are also in the other array
        if event_start is None:
            min_index_data = 0
        else:
            min_index_data = np.argmin(event_number < event_start)

        if event_stop is None:
            max_index_data = event_number.shape[0]
        else:
            max_index_data = np.argmax(event_number >= event_stop)

        if min_index_data < 0:
            min_index_data = 0
        if max_index_data == 0 or max_index_data > event_number.shape[0]:
            max_index_data = event_number.shape[0]
        return array[min_index_data:max_index_data]
    else:
        return array[ne.evaluate('event_number >= event_start & event_number < event_stop')]


def fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=3., n_bad_events=5, n_good_events=3, correlation_search_range=2000, good_events_search_range=10):
    correlated = np.ascontiguousarray(np.ones(shape=event_numbers.shape, dtype=np.uint8))  # array to signal correlation to be ables to omit not correlated events in the analysis
    event_numbers = np.ascontiguousarray(event_numbers)
    ref_column = np.ascontiguousarray(ref_column)
    column = np.ascontiguousarray(column)
    ref_row = np.ascontiguousarray(ref_row)
    row = np.ascontiguousarray(row)
    ref_charge = np.ascontiguousarray(ref_charge, dtype=np.uint16)
    charge = np.ascontiguousarray(charge, dtype=np.uint16)
    n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, correlated, error, n_bad_events, correlation_search_range, n_good_events, good_events_search_range)
    return correlated, n_fixes

if __name__ == '__main__':
    print 'MAIN'
#     with tb.open_file(r'C:\Users\DavidLP\git\pyTestbeamAnalysis\pyTestbeamAnalysis\converter\Tracklets_neu.h5', 'r') as in_file_h5:
#         limit = 40000
#         event_numbers = in_file_h5.root.Tracklets[:limit]['event_number'].copy()
#         ref_column, ref_row = in_file_h5.root.Tracklets[:limit]['column_dut_0'].copy(), in_file_h5.root.Tracklets[:limit]['row_dut_0'].copy()
#         column, row = in_file_h5.root.Tracklets[:limit]['column_dut_1'].copy(), in_file_h5.root.Tracklets[:limit]['row_dut_1'].copy()
#         print in_file_h5.root.Tracklets.dtype.names
#
#         for index, (event, rr, r, cc, c) in enumerate(np.column_stack((event_numbers, ref_row, row, ref_column, column))):
#             if event > 7140 and event < 7180:
#                 print index, int(event), rr, r
#
#         print '___________________'
#
#         corr, n_fixes = fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=2., n_bad_events=30, n_good_events=10, correlation_search_range=100000, good_events_search_range=100)
#
#         print '_____fixes', n_fixes, 'jumps _____'
#
#         for index, (event, rr, r, cc, c, co) in enumerate(np.column_stack((event_numbers, ref_row, row, ref_column, column, corr))):
#             if index > 40000 - 100 and index < 40000:
#                 print index, int(event), rr, r, co
#
#         print '___________________'
#
# #         print event_numbers[-1] - 518
# #
# #         for i in range(287, 500, 1):
# #             print
# #
#         for index, (event, rr, r, cc, c, co) in enumerate(np.column_stack((event_numbers, ref_row, row, ref_column, column, corr))):
#             if event > 7485 and event < 7550:
#                 print index, int(event), rr, r, co
#
#         print '___________________'
#
#         corr = np.logical_and(fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=2., n_bad_events=10), corr)
#
#         for index, (event, rr, r, cc, c, co) in enumerate(np.column_stack((event_numbers, ref_row, row, ref_column, column, corr))):
#             if event > 7485 and event < 7550:
#                 print index, int(event), rr, r, co
#
#         print '___________________'
#
#         corr = fix_event_alignment(event_numbers, ref_column, column, ref_row, row, error=2., n_bad_events=10)
#
#         for index, (event, rr, r, cc, c, co) in enumerate(np.column_stack((event_numbers, ref_row, row, ref_column, column, corr))):
#             if event > 7230 and event < 7280:
#                 print index, int(event), rr, r, co
#
#         print '___________________'

    size = 50
    np.random.seed(0)
    event_numbers = np.arange(size, dtype=np.int64).repeat(2)[:size]
    ref_column, ref_row = np.random.uniform(high=80, size=size), np.random.uniform(high=336, size=size)
    column, row = ref_column.copy(), ref_row.copy()

#     column, row = np.zeros_like(column), np.zeros_like(row)
#     column[0:-10] = ref_column[10:]
#     row[0:-10] = ref_row[10:]

    column[5:16] = 0
    row[5:16] = 0

    column[16:19] = ref_column[6:9]
    row[16:19] = ref_row[6:9]

#     column[20:] = ref_column[20:]
#     row[20:] = ref_row[20:]
#
#     print event_numbers.shape
#     print ref_column.shape
#     print ref_row.shape
#     print column.shape
#     print row.shape

#     event_numbers[13:-1] = event_numbers[14:]
#     ref_column[13:-1] = ref_column[14:]
#     ref_row[13:-1] = ref_row[14:]
#     column[13:-1] = column[14:]
#     row[13:-1] = row[14:]
#
#     column[13], column[14] = column[14], column[13]
#     row[13], row[14] = row[14], row[13]
#
#     column[15], column[16] = column[16], column[15]
#     row[15], row[16] = row[16], row[15]
#
#     column[17], column[18] = column[18], column[17]
#     row[17], row[18] = row[18], row[17]

#     column[19], column[20] = column[20], column[19]
#     row[19], row[20] = row[20], row[19]

#     row[10] = 43.9051
#     row[18] = 43.9051

    corr = np.ascontiguousarray(np.ones(shape=event_numbers.shape, dtype=np.uint8))  # array to signal correlation to be ables to omit not correlated events in the analysis
    corr[17] = 0

    for index, (event, hit_1, hit_2, c) in enumerate(np.column_stack((event_numbers, ref_row, row, corr))):
        print index, int(event), hit_1, hit_2, c
    print '___________________'


    # n_good_events number of correlated events after the first correlated hit candidate within the good_events_search_range

    event_numbers = np.ascontiguousarray(event_numbers)
    ref_column = np.ascontiguousarray(ref_column)
    column = np.ascontiguousarray(column)
    ref_row = np.ascontiguousarray(ref_row)
    row = np.ascontiguousarray(row)
    n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, corr, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=10)

    print n_fixes

    print '___________________'
    for index, (event, hit_1, hit_2, c) in enumerate(np.column_stack((event_numbers, ref_row, row, corr))):
        print index, int(event), hit_1, hit_2, c
    print '___________________'

    print (ref_column[np.logical_and(corr == 1, column != 0)] == column[np.logical_and(corr == 1, column != 0)])

    print 'DONE'
