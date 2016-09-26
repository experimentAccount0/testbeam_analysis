from os.path import join

import tables as tb
import numpy as np

from testbeam_analysis.tools import geometry_utils

class dut(object):
    def __init__(self, id, translation=None, rotation=None, alignment_file=None, name=None):
        self._id = id  # should not be changed
        self.name = '' if name is None else name
        self.init_alignment_array(translation=translation, rotation=rotation)
        self.alignment_file = alignment_file

    @property
    def id(self):
        return self._id

    @property
    def alignment(self):
        return self._alignment_array.copy()[0]

    @property
    def alignment_array(self):
        return self._alignment_array.copy()

    @alignment_array.setter
    def alignment_array(self, alignment_array):
        self._alignment_array[0] = alignment_array[0]
        
    @property
    def alignment_file(self):
        return self._alignment_file

    @alignment_file.setter
    def alignment_file(self, alignment_file):
        self._alignment_file = alignment_file
        if alignment_file is not None:
            self.read_alignment_file()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        
    def init_alignment_array(self, translation=None, rotation=None):
        description = []
        description.append(('translation_x', np.float64))
        description.append(('translation_y', np.float64))
        description.append(('translation_z', np.float64))
        description.append(('alpha', np.float64))
        description.append(('beta', np.float64))
        description.append(('gamma', np.float64))
        alignment_array = np.zeros((1,), dtype=description)
        if translation is None:
            translation = np.array((0, 0, 0), dtype=np.float64)
        else:
            translation = np.array(translation, dtype=np.float64)
        if rotation is None:
            rotation = np.array((0, 0, 0), dtype=np.float64)
        else:
            rotation = np.array(rotation, dtype=np.float64)
        alignment_array["translation_x"] = translation[0]
        alignment_array["translation_y"] = translation[1]
        alignment_array["translation_z"] = translation[2]
        alignment_array["alpha"] = rotation[0]
        alignment_array["beta"] = rotation[1]
        alignment_array["gamma"] = rotation[2]
        self._alignment_array = alignment_array

    def read_alignment_file(self, alignment_file=None):
        if alignment_file is None:
            alignment_file = self.alignment_file
        #else:
        #    self._alignment_file = alignment_file
        
        print "open alignment file", alignment_file
        with tb.open_file(alignment_file, mode="r+") as in_alignment_file:
            try:
                node = in_alignment_file.get_node("/DUT%d" % self.id)
            except tb.NodeError:  # node already exists
                raise ValueError("Node %s does not exist" % ("DUT%d" % self.id,))
            try:
                self.alignment_array = node[0]
            except ValueError:  # node existing but empty
                raise ValueError("Empty alignment data")
            self.read_attributes(node=node)


    def write_alignment_file(self, alignment_file=None):
        if alignment_file is None:
            alignment_file = self.alignment_file
        #else:
        #    self._alignment_file = alignment_file
 
        with tb.open_file(alignment_file, mode="a") as in_alignment_file:
            try:
                node = in_alignment_file.get_node("/DUT%d" % self.id)
            except tb.NodeError:  # node does not exist
                pass
            else:
                node.remove()

            alignment_table = in_alignment_file.create_table(in_alignment_file.root, name="DUT%d" % self.id, title='Alignment parameters for DUT%d%s' % (self.id, (" (" + self.name + ")") if self.name else ""), description=self.alignment_array.dtype)
            node = in_alignment_file.get_node("/DUT%d" % self.id)
            node.append(self.alignment_array)
            self.write_attributes(node=node)
 
    def write_attributes(self, node):
        node._v_attrs.id = self.id
        node._v_attrs.name = self.name

    def read_attributes(self, node):
        self.name = node._v_attrs.name
    
    def index_to_position(self, index):
        raise NotImplementedError
    
    def position_to_index(self, position):
        raise NotImplementedError


class rectangular_pixel_dut(dut):
    def __init__(self, id, column_size, row_size, n_columns, n_rows, translation=None, rotation=None, alignment_file=None, name=None):
        super(rectangular_pixel_dut, self).__init__(id=id, translation=translation, rotation=rotation, alignment_file=alignment_file, name=name)
        self._column_size = column_size
        self._row_size = row_size
        self._n_columns = n_columns
        self._n_rows = n_rows

    @property
    def column_size(self):
        return self._column_size

    @property
    def row_size(self):
        return self._row_size

    @property
    def n_columns(self):
        return self._n_columns

    @property
    def n_rows(self):
        return self._n_rows

    @property
    def x_size(self):
        raise NotImplementedError
        return self._x_size  # TODO: projection onto axis

    @property
    def y_size(self):
        raise NotImplementedError
        return self._y_size  # TODO: projection onto axis

    @property
    def n_x(self):
        raise NotImplementedError
        return self._n_x  # TODO: projection onto axis

    @property
    def n_y(self):
        raise NotImplementedError
        return self._n_y  # TODO: projection onto axis

    def write_attributes(self, node):
        super(rectangular_pixel_dut, self).write_attributes(node=node)
        node._v_attrs.column_size = self.column_size
        node._v_attrs.row_size = self.row_size
        node._v_attrs.n_columns = self.n_columns
        node._v_attrs.n_rows = self.n_rows

    def read_attributes(self, node):
        super(rectangular_pixel_dut, self).read_attributes(node=node)
        self._column_size = node._v_attrs.column_size
        self._row_size = node._v_attrs.row_size
        self._n_columns = node._v_attrs.n_columns
        self._n_rows = node._v_attrs.n_rows

    def index_to_position(self, column, row):
        column = np.array(column, dtype=np.float64)
        row = np.array(row, dtype=np.float64)
        # from index to local coordinates
        x = np.empty_like(column, dtype=np.float64)
        y = np.empty_like(column, dtype=np.float64)
        z = np.full_like(column, fill_value=0.0, dtype=np.float64)  # all DUTs have their origin in 0, 0, 0
        # whether hit index or cluster index is out of range
        hit_selection = np.logical_and(
                            np.logical_and(column >= 0.5, column < self.n_columns + 0.5),
                            np.logical_and(row >= 0.5, row < self.n_rows + 0.5))
        x[hit_selection] = self.column_size * (column[hit_selection] - 0.5 - (0.5 * self.n_columns))
        y[hit_selection] = self.row_size * (row[hit_selection] - 0.5 - (0.5 * self.n_rows))
        x[~hit_selection] = np.nan
        y[~hit_selection] = np.nan
        z[~hit_selection] = np.nan
        # apply DUT alignment
        transformation_matrix = geometry_utils.local_to_global_transformation_matrix(
            x=self.alignment['translation_x'],
            y=self.alignment['translation_y'],
            z=self.alignment['translation_z'],
            alpha=self.alignment['alpha'],
            beta=self.alignment['beta'],
            gamma=self.alignment['gamma'])
        x, y, z = geometry_utils.apply_transformation_matrix(
            x=x,
            y=y,
            z=z,
            transformation_matrix=transformation_matrix)
        return x, y, z

    def position_to_index(self, x, y, z):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        z = np.array(z, dtype=np.float64)
        # apply DUT inverse alignment
        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(
            x=self.alignment['translation_x'],
            y=self.alignment['translation_y'],
            z=self.alignment['translation_z'],
            alpha=self.alignment['alpha'],
            beta=self.alignment['beta'],
            gamma=self.alignment['gamma'])
        x, y, z = geometry_utils.apply_transformation_matrix(
            x=x,
            y=y,
            z=z,
            transformation_matrix=transformation_matrix)
        if not np.allclose(np.nan_to_num(z), 0.0):
            raise RuntimeError('The transformation to the local coordinate system did not give all z = 0.')
        column = np.empty_like(x, dtype=np.float64)
        row = np.empty_like(x, dtype=np.float64)
        column = (x / self.column_size) + 0.5 + (0.5 * self.n_columns)
        row = (y / self.row_size) + 0.5 + (0.5 * self.n_rows)
        # remove nans
        column = np.nan_to_num(column)
        row = np.nan_to_num(row)
        if np.any(np.logical_and(
            np.logical_and(
                column != 0.0,
                row != 0.0),
            np.logical_or(
                np.logical_or(column < 0.5, column >= (self.n_columns + 0.5)),
                np.logical_or(row < 0.5, row >= (self.n_rows + 0.5))))):
            raise RuntimeError('The transformation to the local coordinate system did not give all columns and rows within boundaries.')
        return column, row
