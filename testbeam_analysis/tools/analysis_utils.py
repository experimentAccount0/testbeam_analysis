"""This class provides often needed analysis functions, for analysis that is done with python.
"""
from __future__ import division

import logging
import numpy as np
import numexpr as ne
import tables as tb
from numba import njit
from scipy.interpolate import splrep, sproot
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import quad

from testbeam_analysis import analysis_functions
import testbeam_analysis.tools.plot_utils
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

        while event_index_data_2 < data_2.shape[0] and data_2[event_index_data_2]['event_number'] < data_1[index_data_1]['event_number']:  # Catch up with outer loop
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


@njit
def correlate_cluster_on_event_number(data_1, data_2, column_corr_hist, row_corr_hist):
    """Correlating the hit/cluster indices of two arrays on an event basis with all permutations.
    In other words: correlate all hit/cluster indices of particular event in data_2 with all hit/cluster indices of the same event in data_1.
    Then the hit/cluster indices are used to fill a correlation histograms.

    Does the same than the merge of the pandas package:
        df = data_1.merge(data_2, how='left', on='event_number')
        df.dropna(inplace=True)
        correlation_column = np.hist2d(df[column_mean_dut_0], df[column_mean_dut_x])
        correlation_row = np.hist2d(df[row_mean_dut_0], df[row_mean_dut_x])
    The following code is > 10x faster than the above code.

    Parameters
    ----------
    data_1, data_2: np.recarray
        Hit/cluster array. Must have event_number / mean_column / mean_row columns.
    column_corr_hist, row_corr_hist: np.array
        2D correlation array containing the correlation data. Has to be of sufficient size.

    """
    index_data_2 = 0

    # Loop to determine the needed result array size
    for index_data_1 in range(data_1.shape[0]):

        while index_data_2 < data_2.shape[0] and data_2[index_data_2]['event_number'] < data_1[index_data_1]['event_number']:  # Catch up with outer loop
            index_data_2 += 1

        for event_index_data_2 in range(index_data_2, data_2.shape[0]):
            if data_1[index_data_1]['event_number'] == data_2[event_index_data_2]['event_number']:
                # Assuming value is an index, cluster index 1 from 0.5 to 1.4999, index 2 from 1.5 to 2.4999, etc.
                column_index_dut_1 = int(np.floor(data_1[index_data_1]['mean_column'] - 0.5))
                row_index_dut_1 = int(np.floor(data_1[index_data_1]['mean_row'] - 0.5))
                column_index_dut_2 = int(np.floor(data_2[event_index_data_2]['mean_column'] - 0.5))
                row_index_dut_2 = int(np.floor(data_2[event_index_data_2]['mean_row'] - 0.5))

                if not (column_index_dut_1 >= 0 and row_index_dut_1 >= 0 and column_index_dut_2 >= 0 and row_index_dut_2 >= 0):
                    raise ValueError('Column and/or row index is smaller than 0.5')

                # Add correlation to histogram
                column_corr_hist[column_index_dut_2, column_index_dut_1] += 1
                row_corr_hist[row_index_dut_2, row_index_dut_1] += 1
            else:
                break


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
    event_result = np.empty(shape=(events_one.shape[0] + events_two.shape[0],), dtype=events_one.dtype)
    count = analysis_functions.get_max_events_in_both_arrays(events_one, events_two, event_result)
    return event_result[:count]


def map_cluster(events, cluster):
    """
    Maps the cluster hits on events. Not existing cluster in events have all values set to 0 and column/row/charge set to nan.
    Too many cluster per event for the event number are omitted and lost!

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
    mapped_cluster = np.zeros((events.shape[0],), dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
    mapped_cluster['mean_column'] = np.nan
    mapped_cluster['mean_row'] = np.nan
    mapped_cluster['charge'] = np.nan
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
            if event_number[0] > event_start:
                min_index_data = 0
            else:
                min_index_data = np.argmin(event_number < event_start)

        if event_stop is None:
            max_index_data = event_number.shape[0]
        else:
            if event_number[-1] < event_stop:
                max_index_data = event_number.shape[0]
            else: 
                max_index_data = np.argmax(event_number >= event_stop)

        if min_index_data < 0:
            min_index_data = 0
        return array[min_index_data:max_index_data]
    else:
        return array[ne.evaluate('event_number >= event_start & event_number < event_stop')]


def data_aligned_at_events(table, start_event_number=None, stop_event_number=None, start_index=None, stop_index=None, chunk_size=10000000, try_speedup=False, first_event_aligned=True, fail_on_missing_events=True):
    '''Takes the table with a event_number column and returns chunks with the size up to chunk_size. The chunks are chosen in a way that the events are not splitted.
    Additional parameters can be set to increase the readout speed. Events between a certain range can be selected.
    Also the start and the stop indices limiting the table size can be specified to improve performance.
    The event_number column must be sorted.
    In case of try_speedup is True, it is important to create an index of event_number column with pytables before using this function. Otherwise the queries are slowed down.

    Parameters
    ----------
    table : pytables.table
        The data.
    start_event_number : int
        The retruned data contains events with event number >= start_event_number. If None, no limit is set.
    stop_event_number : int
        The retruned data contains events with event number < stop_event_number. If None, no limit is set.
    start_index : int
        Start index of data. If None, no limit is set.
    stop_index : int
        Stop index of data. If None, no limit is set.
    chunk_size : int
        Maximum chunk size per read.
    try_speedup : bool
        If True, try to reduce the index range to read by searching for the indices of start and stop event number. If these event numbers are usually
        not in the data this speedup can even slow down the function!

    The following parameters are not used when try_speedup is True:

    first_event_aligned : bool
        If True, assuming that the first event is aligned to the data chunk and will be added. If False, the lowest event number of the first chunk will not be read out.
    fail_on_missing_events : bool
        If True, an error is given when start_event_number or stop_event_number is not part of the data.

    Returns
    -------
    Iterator of tuples
        Data of the actual data chunk and start index for the next chunk.

    Example
    -------
    start_index = 0
    for scan_parameter in scan_parameter_range:
        start_event_number, stop_event_number = event_select_function(scan_parameter)
        for data, start_index in data_aligned_at_events(table, start_event_number=start_event_number, stop_event_number=stop_event_number, start_index=start_index):
            do_something(data)

    for data, index in data_aligned_at_events(table):
        do_something(data)
    '''
    # initialize variables
    start_index_known = False
    stop_index_known = False
    start_index = 0 if start_index is None else start_index
    stop_index = table.nrows if stop_index is None else stop_index
    if stop_index < start_index:
        raise ValueError('Invalid start/stop index')
    table_max_rows = table.nrows
    if stop_event_number is not None and start_event_number is not None and stop_event_number < start_event_number:
        raise ValueError('Invalid start/stop event number')

    # set start stop indices from the event numbers for fast read if possible; not possible if the given event number does not exist in the data stream
    if try_speedup and table.colindexed["event_number"]:
        if start_event_number is not None:
            start_condition = 'event_number==' + str(start_event_number)
            start_indices = table.get_where_list(start_condition, start=start_index, stop=stop_index)
            if start_indices.shape[0] != 0:  # set start index if possible
                start_index = start_indices[0]
                start_index_known = True

        if stop_event_number is not None:
            stop_condition = 'event_number==' + str(stop_event_number)
            stop_indices = table.get_where_list(stop_condition, start=start_index, stop=stop_index)
            if stop_indices.shape[0] != 0:  # set the stop index if possible, stop index is excluded
                stop_index = stop_indices[0]
                stop_index_known = True

    if start_index_known and stop_index_known and start_index + chunk_size >= stop_index:  # special case, one read is enough, data not bigger than one chunk and the indices are known
        yield table.read(start=start_index, stop=stop_index), stop_index
    else:  # read data in chunks, chunks do not divide events, abort if stop_event_number is reached

        # search for begin
        current_start_index = start_index
        if start_event_number is not None:
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size, stop_index)
                array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
                last_event_in_chunk = array_chunk["event_number"][-1]

                if last_event_in_chunk < start_event_number:
                    current_start_index = current_start_index + chunk_size  # not there yet, continue to next read (assuming sorted events)
                else:
                    first_event_in_chunk = array_chunk["event_number"][0]
#                     if stop_event_number is not None and first_event_in_chunk >= stop_event_number and start_index != 0 and start_index == current_start_index:
#                         raise ValueError('The stop event %d is missing. Change stop_event_number.' % stop_event_number)
                    if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                        raise ValueError('Chunk size too small. Increase chunk size to fit full event.')

                    if not first_event_aligned and first_event_in_chunk == start_event_number and start_index != 0 and start_index == current_start_index:  # first event in first chunk not aligned at index 0, so take next event
                        if fail_on_missing_events:
                            raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
                        chunk_start_index = np.searchsorted(array_chunk["event_number"], start_event_number + 1, side='left')
                    elif fail_on_missing_events and first_event_in_chunk > start_event_number and start_index == current_start_index:
                        raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
                    elif first_event_aligned and first_event_in_chunk == start_event_number and start_index == current_start_index:
                        chunk_start_index = 0
                    else:
                        chunk_start_index = np.searchsorted(array_chunk["event_number"], start_event_number, side='left')
                        if fail_on_missing_events and array_chunk["event_number"][chunk_start_index] != start_event_number and start_index == current_start_index:
                            raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
#                     if fail_on_missing_events and ((start_index == current_start_index and chunk_start_index == 0 and start_index != 0 and not first_event_aligned) or array_chunk["event_number"][chunk_start_index] != start_event_number):
#                         raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
                    current_start_index = current_start_index + chunk_start_index  # calculate index for next loop
                    break
        elif not first_event_aligned and start_index != 0:
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size, stop_index)
                array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
                first_event_in_chunk = array_chunk["event_number"][0]
                last_event_in_chunk = array_chunk["event_number"][-1]

                if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                    raise ValueError('Chunk size too small. Increase chunk size to fit full event.')

                chunk_start_index = np.searchsorted(array_chunk["event_number"], first_event_in_chunk + 1, side='left')
                current_start_index = current_start_index + chunk_start_index
                if not first_event_in_chunk == last_event_in_chunk:
                    break

        # data loop
        while current_start_index < stop_index:
            current_stop_index = min(current_start_index + chunk_size, stop_index)
            array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
            first_event_in_chunk = array_chunk["event_number"][0]
            last_event_in_chunk = array_chunk["event_number"][-1]

            chunk_start_index = 0

            if stop_event_number is None:
                if current_stop_index == table_max_rows:
                    chunk_stop_index = array_chunk.shape[0]
                else:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], last_event_in_chunk, side='left')
            else:
                if last_event_in_chunk >= stop_event_number:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], stop_event_number, side='left')
                elif current_stop_index == table_max_rows:  # this will also add the last event of the table
                    chunk_stop_index = array_chunk.shape[0]
                else:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], last_event_in_chunk, side='left')

            nrows = chunk_stop_index - chunk_start_index
            if nrows == 0:
                if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                    raise ValueError('Chunk size too small to fit event. Data corruption possible. Increase chunk size to read full event.')
                elif chunk_start_index == 0:  # not increasing current_start_index
                    return
                elif stop_event_number is not None and last_event_in_chunk >= stop_event_number:
                    return
            else:
                yield array_chunk[chunk_start_index:chunk_stop_index], current_start_index + nrows + chunk_start_index

            current_start_index = current_start_index + nrows + chunk_start_index  # events fully read, increase start index and continue reading


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


def find_closest(arr, values):
    '''Returns a list of indices with values closest to arr values.

    Parameters
    ----------
    arr : iterable
        Iterable of numbers. Arr must be sorted.
    values : iterable
        Iterable of numbers.

    Returns
    -------
    A list of indices with values closest to arr values.

    See also: http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    '''
    idx = arr.searchsorted(values)
    idx = np.clip(idx, 1, len(arr)-1)
    left = arr[idx-1]
    right = arr[idx]
    idx -= values - left < right - values
    return idx


def linear(x, c0, c1):
    return c0 + c1 * x


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))


def gauss2(x, *p):
    mu, sigma = p
    return (sigma * np.sqrt(2.0 * np.pi))**-1.0 * np.exp(-0.5 * ((x - mu) / sigma)**2.0)


def gauss_offset_slope(x, *p):
    A, mu, sigma, offset, slope = p
    return gauss(x, A, mu, sigma) + offset + x * slope


def gauss_offset(x, *p):
    A, mu, sigma, offset = p
    return gauss(x, A, mu, sigma) + offset


def double_gauss(x, *p):
    A_1, mu_1, sigma_1, A_2, mu_2, sigma_2 = p
    return gauss(x, A_1, mu_1, sigma_1) + gauss(x, A_2, mu_2, sigma_2)


def double_gauss_offset(x, *p):
    A_1, mu_1, sigma_1, A_2, mu_2, sigma_2, offset = p
    return gauss(x, A_1, mu_1, sigma_1) + gauss(x, A_2, mu_2, sigma_2) + offset


def gauss_box(x, *p):
    ''''Convolution of gaussian and rectangle is a gaussian integral.

    Parameters
    ----------
    A, mu, sigma, a (width of the rectangle) : float

    See also: http://stackoverflow.com/questions/24230233/fit-gaussian-integral-function-to-data
    '''
    A, mu, sigma, a = p
    return quad(lambda t: gauss(x-t,A,mu,sigma),-a,a)[0]

# Vetorize function to use with np.arrays
gauss_box_vfunc = np.vectorize(gauss_box, excluded=["*p"])


def get_chi2(y_data, y_fit):
    return np.square(y_data - y_fit).sum()


def get_mean_from_histogram(counts, bin_positions):
    return np.dot(counts, np.array(bin_positions)) / np.sum(counts).astype('f4')


def get_rms_from_histogram(counts, bin_positions):
    return np.std(np.repeat(bin_positions, counts))


def get_median_from_histogram(counts, bin_positions):
    return np.median(np.repeat(bin_positions, counts))


def get_mean_efficiency(array_pass, array_total, method=0):
    ''' Function to calculate the mean and the error of the efficiency using different approaches.
    No good approach was found.

    Parameters
    ---------
    array_pass : numpy array
        Array with tracks that were seen by the DUT. Can be a masked array.
    array_total : numpy array
        Array with total tracks in the DUT. Can be a masked array.
    method: number
        Select the method to calculate the efficiency:
          1: Take mean and RMS of the efficiency per bin. Each bin is weightd by the number of total tracks.
             Results in symmetric unphysical error bars that cover also efficiencies > 100%
          2: Takes a correct binomial distribution and calculate the correct confidence interval after combining all pixels.
             Gives correct mean and correct statistical error bars. Systematic efficiency variations are not taken into account, thus
             the error bar is very small. Further info: https://root.cern.ch/doc/master/classTEfficiency.html
          3: As 2. but for each efficiency bin separately. Then the distribution is fitted by a constant using the
             asymmetric error bars. Gives too high efficiencies and small,  asymmetric error bars.
          4: Use a special Binomial efficiency fit. Gives best mean efficiency but unphysical symmetric error
             bars. Further info: https://root.cern.ch/doc/master/classTBinomialEfficiencyFitter.html
    '''
    
    n_bins = np.ma.count(array_pass)
    logging.info('Calculate the mean efficiency from %d pixels', n_bins)
    
    if method == 0:
        def weighted_avg_and_std(values, weights):  # http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
            average = np.average(values, weights=weights)
            variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
            return (average, np.sqrt(variance))

        efficiency = np.ma.compressed(array_pass) / np.ma.compressed(array_total)
        return weighted_avg_and_std(efficiency, np.ma.compressed(array_total))
    else:  # Use CERN ROOT
        try:
            from ROOT import TH1D, TEfficiency, TF1, TBinomialEfficiencyFitter
        except ImportError:
            raise RuntimeError('To use these method you have to install CERN ROOT with python bindings.')
    
        # Convert not masked numpy array values to 1D ROOT double histogram
        def fill_1d_root_hist(array, name):
            length = np.ma.count(array_pass)
            root_hist = TH1D(name, "hist", length, 0, length)
            for index, value in enumerate(np.ma.compressed(array).ravel()):
                root_hist.SetBinContent(index, value)
            return root_hist
        
        if method == 1:
            # The following combines all pixel and gives correct
            # statistical errors but does not take the systematic
            # variations of within the  efficiency histogram parts into account
            # Thus gives very small error bars
            hist_pass = TH1D("hist_pass", "hist", 1, 0, 1)
            hist_total = TH1D("hist_total", "hist", 1, 0, 1)
            hist_pass.SetBinContent(0, np.ma.sum(array_pass))
            hist_total.SetBinContent(0, np.ma.sum(array_total))
            efficiency = TEfficiency(hist_pass, hist_total)
            return efficiency.GetEfficiency(0), efficiency.GetEfficiencyErrorLow(0), efficiency.GetEfficiencyErrorUp(0)
        elif method == 2:
            # The following fits the efficiency with a constant but
            # it gives symmetric error bars, thus unphysical results
            # This is not understood yet and thus not used
            
            # Convert numpy array to ROOT hists
            hist_pass = fill_1d_root_hist(array_pass, 'h_pass')
            hist_total = fill_1d_root_hist(array_total, 'h_total')
            f1 = TF1("f1", "pol0(0)", 0, n_bins)
            fitter = TBinomialEfficiencyFitter(hist_pass, hist_total)
            r = fitter.Fit(f1, "SE")
            eff_err_low = r.LowerError(0)
            eff_err_up = r.UpperError(0)
            efficiency = r.GetParams()[0]
            return efficiency, eff_err_low, eff_err_up
        elif method == 3:
            # Fit point of each efficiency bin using asymmetric error bars
            # Parameters described here: https://root.cern.ch/doc/master/classTGraph.html#a61269bcd47a57296f0f1d57ceff8feeb
            # This gives too high efficiency and too small error

            # Convert numpy array to ROOT hists
            hist_pass = fill_1d_root_hist(array_pass, 'h_pass')
            hist_total = fill_1d_root_hist(array_total, 'h_total')
            efficiency = TEfficiency(hist_pass, hist_total)
            f1 = TF1("f1", "pol0(0)", 0, n_bins)
            fit_result = efficiency.CreateGraph().Fit(f1, "SFEM")
            eff_mean = fit_result.GetParams()[0]
            eff_err_low = fit_result.LowerError(0)
            eff_err_up = fit_result.UpperError(0)
            return eff_mean, eff_err_low, eff_err_up


def fwhm(x, y):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset. The function
    uses a spline interpolation of order 3.

    See also http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    """
    half_max = np.max(y) / 2.0
    spl = splrep(x, y - half_max)
    roots = sproot(spl)

    if len(roots) != 2:  # multiple peaks or no peaks
        raise RuntimeError("Cannot determine FWHM")
    else:
        return roots[0], roots[1]


def peak_detect(x, y):
    try:
        fwhm_left_right = fwhm(x=x, y=y)
    except (RuntimeError, TypeError):
        raise RuntimeError("Cannot determine peak")
    fwhm_value = fwhm_left_right[-1] - fwhm_left_right[0]
    max_position = x[np.argmax(y)]
    center = (fwhm_left_right[0] + fwhm_left_right[-1]) / 2.0
    return max_position, center, fwhm_value, fwhm_left_right


def simple_peak_detect(x, y):
    y = np.array(y)
    half_maximum = np.max(y) * 0.5
    greater = (y > half_maximum)
    change_indices = np.where(greater[:-1] != greater[1:])[0]
    if np.all(greater == False) or greater[0] == True or greater[-1] == True:
        raise RuntimeError("Cannot determine peak")
    x = np.array(x)
    # get center of indices for higher precision peak position and FWHM
    x_center = (x[1:] + x[:-1]) / 2.0
    fwhm_left_right = (x_center[change_indices[0]], x_center[change_indices[-1]])
    fwhm_value = fwhm_left_right[-1] - fwhm_left_right[0]
    max_position = x[np.argmax(y)]
    center = (fwhm_left_right[0] + fwhm_left_right[-1]) / 2.0
    return max_position, center, fwhm_value, fwhm_left_right


def get_rotation_from_residual_fit(m_xx, m_xy, m_yx, m_yy, alpha_inverted=None, beta_inverted=None):
    if np.abs(m_xy) > 1. or np.abs(m_yx) > 1.:
        raise NotImplementedError('Device seems to be heavily tilted in gamma. This is not supported.')

    # Detect device rotation around y-axis (beta angle)
    if m_yy < -1. or alpha_inverted:
        logging.info('Device most likely inverted in beam around the x axis (y coordinates switched)!')
        alpha_inverted = True
        m_yy = 2 + m_yy
    else:
        alpha_inverted = False

    # Detect device rotation around y-axis (beta angle)
    if m_xx < -1.:
        logging.info('Device most likely inverted in beam around the y axis (x coordinates switched)!')
        beta_inverted = True
        m_xx = 2 + m_xx
    else:
        beta_inverted = False

    # Sanity checks
    if m_xx < -2:
        raise
        m_xx = -2.

    if m_yy < -2:
        raise
        m_yy = -2.

    # Deduce from reidual correlations the rotation matrix
    # gamma (rotation around z)
    # TODO: adding some weighting factor based on fit error
    factor_xy = 1.0
    gamma = factor_xy * np.arctan2(m_xy, 1 - m_xx)
    factor_yx = 1.0
    gamma -= factor_yx * np.arctan2(m_yx, 1 - m_yy)
    gamma /= (factor_xy + factor_yx)

    # cos(gamma) = 1 - m_xx / cos(gamma) = (1 - m_xx) * sqrt(m_xy**2 / (1 - m_xx**2) + 1.) ?
    cosbeta = (1 - m_xx) * np.sqrt(np.square(m_xy) / np.square(1 - m_xx) + 1.)

    # TODO: Why is this needed? Most likely stability reasons
    if np.abs(cosbeta) > 1:
        cosbeta = 1 - (cosbeta - 1)
    beta = np.arccos(cosbeta)

    # cos(alpha) = - myy - tan(gamma) * myx + 1 / (cos(gamma) + sin(gamma) * tan(gamma)) = - myy - m_xy / (1 - m_xx) * myx + 1 / (cos(gamma) + sin(gamma) * tan(gamma)) ?
    cosalpha = (-m_yy - m_xy / (1 - m_xx) * m_yx + 1) / (np.cos(gamma) + np.sin(gamma) * np.tan(gamma))

    # TODO: Why is this needed? Most likely stability reasons
    if np.abs(cosalpha) > 1:
        cosalpha = 1 - (cosalpha - 1)
    alpha = np.arccos(cosalpha)

    if alpha_inverted:
        alpha -= np.pi

        # Try to deduce correct beta sign
        value_1 = -m_xy / (1 + m_xx)
        value_2 = (m_yx - np.sin(alpha) * np.sin(beta) * np.cos(gamma)) / (1 + m_yy - np.sin(alpha) * np.sin(beta) * np.sin(gamma))
        value_3 = (m_yx + np.sin(alpha) * np.sin(beta) * np.cos(gamma)) / (1 + m_yy + np.sin(alpha) * np.sin(beta) * np.sin(gamma))

        if np.abs(value_2 - value_1) > np.abs(value_3 - value_1):
            beta *= -1.

        alpha *= -1.

    if beta_inverted:
        beta -= np.pi

        # Try to deduce correct alpha sign
        value_1 = -m_xy / (1 + m_xx)
        value_2 = (m_yx - np.sin(alpha) * np.sin(beta) * np.cos(gamma)) / (1 + m_yy - np.sin(alpha) * np.sin(beta) * np.sin(gamma))
        value_3 = (m_yx + np.sin(alpha) * np.sin(beta) * np.cos(gamma)) / (1 + m_yy + np.sin(alpha) * np.sin(beta) * np.sin(gamma))

        if np.abs(value_2 - value_1) < np.abs(value_3 - value_1):
            alpha *= -1.
        beta *= -1.

    return alpha, beta, gamma


def fit_residuals(hist, edges, label="", title="", output_pdf=None):
    bin_center = (edges[1:] + edges[:-1]) / 2.0
    hist_mean = get_mean_from_histogram(hist, bin_center)
    hist_std = get_rms_from_histogram(hist, bin_center)
    try:
        fit, cov = curve_fit(gauss, bin_center, hist, p0=[np.amax(hist), hist_mean, hist_std])
    except RuntimeError:
        fit, cov = [np.amax(hist), hist_mean, hist_std], np.full((3, 3), np.nan)

    testbeam_analysis.tools.plot_utils.plot_residuals(
        histogram=hist,
        edges=edges,
        fit=fit,
        fit_errors=cov,
        x_label=label,
        title=title,
        output_pdf=output_pdf
    )

    return fit, cov


def fit_residuals_vs_position(hist, xedges, yedges, xlabel="", ylabel="", title="", output_pdf=None):
    xcenter = (xedges[1:] + xedges[:-1]) / 2.0
    ycenter = (yedges[1:] + yedges[:-1]) / 2.0
    y_sum = np.sum(hist, axis=1)
    x_sel = (y_sum > 0.0) & np.isfinite(y_sum)
    y_mean = np.full_like(y_sum, np.nan, dtype=np.float)
    y_mean[x_sel] = np.average(hist, axis=1, weights=ycenter)[x_sel] * np.sum(ycenter) / y_sum[x_sel]
    n_hits_threshold = np.percentile(y_sum, 100 - 68)
    x_sel = (y_sum > n_hits_threshold) & np.isfinite(y_sum)
    y_rel_err = np.full_like(y_sum, np.nan, dtype=np.float)
    y_rel_err[x_sel] = np.sum(y_sum[x_sel]) / y_sum[x_sel]
    fit, cov = curve_fit(linear, xcenter[x_sel], y_mean[x_sel], sigma=y_rel_err[x_sel], absolute_sigma=False)

    testbeam_analysis.tools.plot_utils.plot_residuals_vs_position(
        hist,
        xedges=xedges,
        yedges=yedges,
        xlabel=xlabel,
        ylabel=ylabel,
        res_mean=y_mean,
        res_pos=xcenter,
        selection=x_sel,
        fit=fit,
        cov=cov,
        title=title,
        output_pdf=output_pdf
    )

    return fit, cov


def hough_transform(img, theta_res=1.0, rho_res=1.0, return_edges=False):
    thetas = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1)
    thetas = np.concatenate((thetas, -thetas[len(thetas)-2::-1]))
    thetas = np.deg2rad(thetas)
    width, height = img.shape
    diag_len = np.sqrt((width - 1)**2 + (height - 1)**2)
    q = np.ceil(diag_len/rho_res)
    nrhos = 2 * q + 1
    rhos = np.linspace(-q * rho_res, q * rho_res, nrhos)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((rhos.size, thetas.size), dtype=np.int)
    y_idxs, x_idxs = np.nonzero(img)

    @njit
    def loop(accumulator, x_idxs, y_idxs, thetas, rhos, sin_t, cos_t):

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for theta_idx in range(thetas.size):
                #rho_idx = np.around(x * cos_t[theta_idx] + y * sin_t[theta_idx]) + diag_len
                rhoVal = x * cos_t[theta_idx] + y * sin_t[theta_idx]
                rho_idx = (np.abs(rhos - rhoVal)).argmin()
                accumulator[rho_idx, theta_idx] += 1
    loop(accumulator, x_idxs, y_idxs, thetas, rhos, sin_t, cos_t)

    if return_edges:
        thetas_diff = thetas[1] - thetas[0]
        thetas_edges = (thetas[1:] + thetas[:-1]) / 2.0
        theta_edges = np.r_[thetas_edges[0] - thetas_diff, thetas_edges, thetas_edges[-1] + thetas_diff]
        rho_diff = rhos[1] - rhos[0]
        rho_edges = (rhos[1:] + rhos[:-1]) / 2.0
        rho_edges = np.r_[rho_edges[0] - rho_diff, rho_edges, rho_edges[-1] + rho_diff]
        return accumulator, thetas, rhos, theta_edges, rho_edges  # return histogram, bin centers, edges
    else:
        return accumulator, thetas, rhos  # return histogram and bin centers
