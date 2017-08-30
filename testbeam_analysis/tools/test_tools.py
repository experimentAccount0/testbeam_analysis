''' Helper functions for the unittests are defined here.
'''

import logging
import os
import inspect
import itertools

import numpy as np
import tables as tb

FIXTURE_FOLDER = 'fixtures'


def nan_to_num(array):
    ''' Like np.nan_to_num but also works on recarray
    '''
    if array.dtype.names is None:  # normal nd.array
        array = np.nan_to_num(array)
    else:
        for column_name in array.dtype.names:
            array[column_name] = np.nan_to_num(array[column_name])


def get_array_differences(first_array, second_array):
    '''Takes two numpy.ndarrays and compares them on a column basis.
    Different column data types, missing columns and columns with different values are returned in a string.

    Parameters
    ----------
    first_array : numpy.ndarray
    second_array : numpy.ndarray

    Returns
    -------
    string
    '''
    if first_array.dtype.names is None:  # normal nd.array
        return ': Sum first array: ' + str(np.sum(first_array)) + ', Sum second array: ' + str(np.sum(second_array))
    else:
        return_str = ''
        for column_name in first_array.dtype.names:
            first_column = first_array[column_name]
            nan_to_num(first_column)  # Otherwise check with nan fails
            try:
                second_column = second_array[column_name]
                nan_to_num(second_column)  # Otherwise check with nan fails
            except ValueError:
                return_str += 'No ' + column_name + ' column found. '
                continue
            if (first_column.dtype != second_column.dtype):
                return_str += 'Column ' + column_name + ' has different data type. '
            if (first_column.shape != second_column.shape):
                return_str += 'The array length is different: %s != %s' % (str(first_column.shape), str(second_column.shape))
                return ': ' + return_str
            if not np.all(first_column == second_column):  # Check if the data of the column is equal
                return_str += 'Column ' + column_name + ' not equal'
                if np.allclose(first_column, second_column, rtol=1.e-5, atol=1.e-8, equal_nan=True):
                    return_str += ', but close'
                return_str += '. '
        for column_name in second_array.dtype.names:
            try:
                first_array[column_name]
            except ValueError:
                return_str += 'Additional column ' + column_name + ' found. '
                continue
        return ': ' + return_str


def array_close(array_1, array_2, rtol=1.e-5, atol=1.e-8):
    '''Compares two numpy arrays elementwise for similarity with small differences.'''
    if not array_1.dtype.names:  # Not a recarray
        try:
            return np.allclose(array_1, array_2, rtol=1.e-5, atol=1.e-8)  # Only works on non recarrays
        except ValueError:  # Raised if shape is incompatible
            return False

    # Check if same data fields
    if sorted(array_1.dtype.names) != sorted(array_2.dtype.names):
        return False
    results = []
    for column in array_1.dtype.names:
        results.append(np.allclose(array_1[column], array_2[column], rtol=1.e-5, atol=1.e-8))
    return np.all(results)


def compare_h5_files(first_file, second_file, expected_nodes=None, detailed_comparison=True, exact=True, rtol=1.e-5, atol=1.e-8):
    '''Takes two hdf5 files and check for equality of all nodes.
    Returns true if the node data is equal and the number of nodes is the number of expected nodes.
    It also returns a error string containing the names of the nodes that are not equal.

    Parameters
    ----------
    first_file : string
        Path to the first file.
    second_file : string
        Path to the first file.
    expected_nodes : Int
        The number of nodes expected in the second_file. If not specified the number of nodes expected in the second_file equals
        the number of nodes in the first file.
    detailed_comparison : boolean
        Print reason why the comparison failed
    exact : boolean
        True if the results have to match exactly. E.g. False for fit results.
    rtol, atol: number
        From numpy.allclose:
        rtol : float
            The relative tolerance parameter (see Notes).
        atol : float
            The absolute tolerance parameter (see Notes).
    Returns
    -------
    bool, string
    '''
    checks_passed = True
    error_msg = ""
    with tb.open_file(first_file, 'r') as first_h5_file:
        with tb.open_file(second_file, 'r') as second_h5_file:
            if expected_nodes is None:
                n_expected_nodes = sum(1 for _ in enumerate(first_h5_file.root))
            else:
                n_expected_nodes = expected_nodes  # set the number of expected nodes
            n_nodes = sum(1 for _ in enumerate(second_h5_file.root))  # calculated the number of nodes
            if n_nodes != n_expected_nodes:
                checks_passed = False
                error_msg += 'The number of nodes in the file is wrong.\n'
            for node in second_h5_file.root:  # loop over all nodes and compare each node, do not abort if one node is wrong
                try:
                    expected_data = first_h5_file.get_node(first_h5_file.root, node.name)[:]
                    data = second_h5_file.get_node(second_h5_file.root, node.name)[:]
                    # Convert nan to number otherwise check fails
                    nan_to_num(data)
                    nan_to_num(expected_data)
                    if exact:
                        # Use close without error to allow equal nan
                        if not np.array_equal(expected_data, data):
                            checks_passed = False
                            error_msg += node.name
                            if detailed_comparison:
                                error_msg += get_array_differences(expected_data, data)
                            error_msg += '\n'
                    else:
                        if not array_close(expected_data, data, rtol, atol):
                            checks_passed = False
                            error_msg += node.name
                            if detailed_comparison:
                                error_msg += get_array_differences(expected_data, data)
                            error_msg += '\n'
                except tb.NoSuchNodeError:
                    checks_passed = False
                    error_msg += 'Unknown node ' + node.name + '\n'
    return checks_passed, error_msg


def _call_function_with_args(function, **kwargs):
    ''' Calls the function with the given kwargs
    and returns the result in a numpy array. All combinations
    of functions arguments in a list are used for multiple
    function calls.'''

    # Create all combinations of arguments from list parameters
    # This is ugly but avoids recursion and does effectively
    # a nested loop of n parameters:
    # for par_1 in pars_1:
    #  for par_2 in pars_2:
    #    ...
    #    for par_n in pars_n:
    #      function(par_1, par_2, ..., par_n)

    call_values = []  # Arguments with permutations
    fixed_arguments = []  # Constant arguments
    fixed_arguments_pos = []
    for index, values in enumerate(kwargs.values()):
        if isinstance(values, list):
            call_values.extend([values])
        else:
            fixed_arguments.append(values)
            fixed_arguments_pos.append(index)
    call_values = list(itertools.product(*call_values))

    data = []

    # Call functions with all parameter combinations
    for call_value in call_values:
        actual_call_value = list(call_value)
        for index, fixed_arg_pos in enumerate(fixed_arguments_pos):
            actual_call_value.insert(fixed_arg_pos, fixed_arguments[index])
        call_args = {
            key: value for key, value in zip(kwargs.keys(), actual_call_value)}
        data.append(function(**call_args))

    return data


def create_fixture(function, **kwargs):
    ''' Calls the function with the given kwargs values and stores the result.

    Numpy arrays are given as one parameter, lists parameters are looped with repeated
    function calls.
    '''

    # Check if all parameters are defined
    func_args = inspect.getargspec(function)[0]
    if not all([a in kwargs for a in func_args]):
        raise RuntimeError('Not all function arguments values defined')

    data = _call_function_with_args(function, **kwargs)

    # Store function return values in compressed pytable array
    data = np.array(data)
    with tb.open_file(os.path.join(FIXTURE_FOLDER, '%s.h5' % str(function.__name__)), 'w') as out_file:
        data_array = out_file.create_carray(out_file.root, name='Data',
                                            title='%s return values' % function.__name__, atom=tb.Atom.from_dtype(data.dtype),
                                            shape=data.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        data_array[:] = data


def check_with_fixture(function, **kwargs):
    ''' Calls the function with the given kwargs values and compares the result with the fixture.

    Numpy arrays are given as one parameter, lists parameters are looped with repeated
    function calls.
    '''

    with tb.open_file(os.path.join(FIXTURE_FOLDER, '%s.h5' % str(function.__name__)), 'r') as in_file:
        data_fixture = in_file.root.Data[:]

    data = _call_function_with_args(function, **kwargs)

    return np.allclose(data_fixture, data)


def install_quilt_data(package, hash):
    ''' Download data package from https://quiltdata.com and install

        package : str
        Package string in USER/PACKAGE format
    '''

    try:
        import quilt
        from quilt.tools import command, store
    except ImportError:
        logging.error('Install quilt to access data packa %s from https://quiltdata.com', package)

    owner, pkg, _ = command._parse_package(package, allow_subpath=True)
    s = store.PackageStore()
    existing_pkg = s.get_package(owner, pkg)

    if existing_pkg:
        return True
    quilt.install(package, hash=hash)
    return s.get_package(owner, pkg)


def get_quilt_data(root, name):
    ''' Access quilt node data by name.

        root: group node that has data node with name
        name: name of node
    '''

    for n, node in root._items():
        if n == name:
            return node._data()

    raise ValueError('Data %s not found', name)