"""This class provides often needed analysis functions, for analysis that is done with python.
"""
from __future__ import division

import logging
import numpy as np
import numexpr as ne
import tables as tb
from numba import njit

from testbeam_analysis import analysis_functions
from testbeam_analysis.cpp import data_struct


@njit
def merge_on_event_number(data_1, data_2):
    """
    Merges the data_2 with data_1 on an event basis with all permutations
    That means: merge all hits of every event in data_2 on all hits of the same event in data_1.

    Does the same than the merge of the pandas package:
        df = data_1.merge(data_2, how='left', on='event_number')
        df.dropna(inplace=True)
    But results in 4 x faster code.

    Parameter
    --------

    data_1, data_2: np.recarray with event_number column

    Returns
    -------

    Tuple np.recarray, np.recarray
        Is the data_1, data_2 array extended by the permutations.

    """
    result_array_size = 0
    event_index_data_2 = 0

    # Loop to determine the needed result array size
    for index_data_1 in range(data_1.shape[0]):

        while event_index_data_2 < data_2.shape[0] and data_2[event_index_data_2]['event_number'] < data_1[index_data_1]['event_number']:
            event_index_data_2 += 1

        for index_data_2 in range(event_index_data_2, data_2.shape[0]):
            if data_1[index_data_1]['event_number'] == data_2[index_data_2]['event_number']:
                result_array_size += 1
            else:
                break

    # Create result array with correct size
    result_1 = np.zeros(shape=(result_array_size,), dtype=data_1.dtype)
    result_2 = np.zeros(shape=(result_array_size,), dtype=data_2.dtype)

    result_index_1 = 0
    result_index_2 = 0
    event_index_data_2 = 0

    for index_data_1 in range(data_1.shape[0]):

        while event_index_data_2 < data_2.shape[0] and data_2[event_index_data_2]['event_number'] < data_1[index_data_1]['event_number']:
            event_index_data_2 += 1

        for index_data_2 in range(event_index_data_2, data_2.shape[0]):
            if data_1[index_data_1]['event_number'] == data_2[index_data_2]['event_number']:
                result_1[result_index_1] = data_1[index_data_1]
                result_2[result_index_2] = data_2[index_data_2]
                result_index_1 += 1
                result_index_2 += 1
            else:
                break

    return result_1, result_2


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
    Maps the cluster hits on events. Not existing cluster in events have all values set to 0. Too many cluster per event for the event number are omitted
    and lost!

    Parameters
    ----------
    events : numpy array
        One dimensional event number array with increasing event numbers.
    cluster : np.recarray
        Recarray with cluster info. The event number is increasing.

    Example
    -------
    event = [ 0  1  1  2  3  3 ]
    cluster.event_number = [ 0  1  2  2  3  4 ]

    gives mapped_cluster.event_number = [ 0  1  0  2  3  0 ]

    Returns
    -------
    Cluster array with given length of the events array.

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

    if x.shape != y.shape:
        raise ValueError('The dimensions in x / y have to match')

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

    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError('The dimensions in x / y / z have to match')

    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    y = np.ascontiguousarray(y.astype(np.int32))
    z = np.ascontiguousarray(z.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint16).ravel()  # ravel hist in c-style, 3D --> 1D
    analysis_functions.hist_3d(x, y, z, shape[0], shape[1], shape[2], result)
    return np.reshape(result, shape)  # rebuilt 3D hist from 1D hist


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


def data_aligned_at_events(table, start_event_number=None, stop_event_number=None, start=None, stop=None, try_speedup=False, chunk_size=10000000):
    '''Takes the table with a event_number column and returns chunks with the size up to chunk_size. The chunks are chosen in a way that the events are not splitted. Additional
    parameters can be set to increase the readout speed. If only events between a certain event range are used one can specify this. Also the start and the
    stop indices for the reading of the table can be specified for speed up.
    It is important to index the event_number with pytables before using this function, otherwise the queries are very slow.

    Parameters
    ----------
    table : pytables.table
    start_event_number : int
        The data read is corrected that only data starting from the start_event number is returned. Lower event numbers are discarded.
    stop_event_number : int
        The data read is corrected that only data up to the stop_event number is returned. The stop_event number is not included.
    try_speedup : bool
        Try to reduce the index range to read by searching for the indices of start and stop event number. If these event numbers are usually
        not in the data this speedup can even slow down the function!
    Returns
    -------
    iterable to numpy.histogram
        The data of the actual chunk.
    stop_index: int
        The index of the last table part already used. Can be used if data_aligned_at_events is called in a loop for speed up.
        Example:
        start_index = 0
        for scan_parameter in scan_parameter_range:
            start_event_number, stop_event_number = event_select_function(scan_parameter)
            for data, start_index in data_aligned_at_events(table, start_event_number=start_event_number, stop_event_number=stop_event_number, start=start_index):
                do_something(data)
    Example
    -------
    for data, index in data_aligned_at_events(table):
        do_something(data)
    '''

    # initialize variables
    start_index_known = False
    stop_index_known = False
    last_event_start_index = 0
    start_index = 0 if start is None else start
    stop_index = table.nrows if stop is None else stop

    if try_speedup and table.colindexed["event_number"]:
        if start_event_number is not None:
            start_condition = 'event_number==' + str(start_event_number)
            start_indeces = table.get_where_list(start_condition, start=start_index, stop=stop_index)
            if start_indeces.shape[0] != 0:  # set start index if possible
                start_index = start_indeces[0]
                start_index_known = True

        if stop_event_number is not None:
            stop_condition = 'event_number==' + str(stop_event_number)
            stop_indeces = table.get_where_list(stop_condition, start=start_index, stop=stop_index)
            if stop_indeces.shape[0] != 0:  # set the stop index if possible, stop index is excluded
                stop_index = stop_indeces[0]
                stop_index_known = True

    if (start_index_known and stop_index_known) and (start_index + chunk_size >= stop_index):  # special case, one read is enough, data not bigger than one chunk and the indices are known
        yield table.read(start=start_index, stop=stop_index), stop_index
    else:  # read data in chunks, chunks do not divide events, abort if stop_event_number is reached
        while(start_index < stop_index):
            src_array = table.read(start=start_index, stop=start_index + chunk_size + 1)  # stop index is exclusive, so add 1
            first_event = src_array["event_number"][0]
            last_event = src_array["event_number"][-1]
            if (start_event_number is not None and last_event < start_event_number):
                start_index = start_index + src_array.shape[0]  # events fully read, increase start index and continue reading
                continue

            last_event_start_index = np.argmax(src_array["event_number"] == last_event)  # get first index of last event
            if last_event_start_index == 0:
                nrows = src_array.shape[0]
                if nrows != 1:
                    logging.warning("Depreciated warning?! Buffer too small to fit event. Possible loss of data. Increase chunk size.")
            else:
                if start_index + chunk_size > stop_index:  # special case for the last chunk read, there read the table until its end
                    nrows = src_array.shape[0]
                else:
                    nrows = last_event_start_index

            if (start_event_number is not None or stop_event_number is not None) and (last_event > stop_event_number or first_event < start_event_number):  # too many events read, get only the selected ones if specified
                selected_rows = get_data_in_event_range(src_array[0:nrows], event_start=start_event_number, event_stop=stop_event_number, assume_sorted=True)
                if len(selected_rows) != 0:  # only return non empty data
                    yield selected_rows, start_index + len(selected_rows)
            else:
                yield src_array[0:nrows], start_index + nrows  # no events specified or selected event range is larger than read chunk, thus return the whole chunk minus the little part for event alignment
            if stop_event_number is not None and last_event > stop_event_number:  # events are sorted, thus stop here to save time
                break
            start_index = start_index + nrows  # events fully read, increase start index and continue reading


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
