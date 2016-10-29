# Copyright 2016 Feather Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++
# cython: embedsignature = True

from libcpp.string cimport string
from libcpp cimport bool as c_bool

cimport cpython
from cython.operator cimport dereference as deref

from libfeather cimport *

import pandas as pd
from feather.compat import pdapi

from numpy cimport ndarray
cimport numpy as cnp
import numpy as np

from feather.compat import frombytes, tobytes, encode_file_path
import six

cnp.import_array()

class FeatherError(Exception):
    pass


cdef extern from "interop.h" namespace "feather::py":
    Status pandas_to_primitive(object ao, PrimitiveArray* out)
    Status pandas_masked_to_primitive(object ao, object mask,
                                      PrimitiveArray* out)
    object get_null_mask(const PrimitiveArray& arr)
    object raw_primitive_to_pandas(const PrimitiveArray& arr)
    object primitive_to_pandas(const PrimitiveArray& arr)
    void set_numpy_nan(object nan)


cdef check_status(const Status& status):
    if status.ok():
        return

    cdef string c_message = status.ToString()
    raise FeatherError(frombytes(c_message))

cdef update_mask_with_datatype_nulls(mask, values):
    datatype_nulls = pd.isnull(values)
    if datatype_nulls.any():
        if mask is None:
            return datatype_nulls
        else:
            return mask | datatype_nulls

set_numpy_nan(np.nan)

cdef class FeatherWriter:
    cdef:
        unique_ptr[TableWriter] writer
        int64_t num_rows

    def __cinit__(self, object name):
        cdef:
            string c_name = encode_file_path(name)

        check_status(TableWriter.OpenFile(c_name, &self.writer))
        self.num_rows = -1

    def close(self):
        if self.num_rows < 0:
            self.num_rows = 0
        self.writer.get().SetNumRows(self.num_rows)
        check_status(self.writer.get().Finalize())

    def write_array(self, object name, object col, object mask=None):
        if self.num_rows >= 0:
            if len(col) != self.num_rows:
                raise ValueError('prior column had a different number of rows')
        else:
            self.num_rows = len(col)

        if pdapi.is_categorical_dtype(col.dtype):
            self.write_category(name, col, mask)
        elif pdapi.is_datetime64_any_dtype(col.dtype):
            self.write_timestamp(name, col, mask)
        else:
            self.write_primitive(name, col, mask)

    cdef write_category(self, name, col, mask):
        cdef:
            string c_name = tobytes(name)
            PrimitiveArray values
            PrimitiveArray levels

        col_values = _unbox_series(col)
        mask_to_file = update_mask_with_datatype_nulls(mask, col_values)

        self.write_ndarray(col_values.codes, mask_to_file, &values)
        check_status(pandas_to_primitive(np.asarray(col_values.categories),
                                         &levels))
        check_status(self.writer.get().AppendCategory(c_name, values, levels,
                                                      col_values.ordered))

    cdef write_primitive(self, name, col, mask):
        cdef:
            string c_name = tobytes(name)
            PrimitiveArray values

        col_values = _unbox_series(col)
        self.write_ndarray(col_values, mask, &values)
        check_status(self.writer.get().AppendPlain(c_name, values))

    cdef write_timestamp(self, name, col, mask):
        cdef:
            string c_name = tobytes(name)
            PrimitiveArray values
            TimestampMetadata metadata

        col_values = _unbox_series(col)
        mask_to_file = update_mask_with_datatype_nulls(mask, col_values)

        self.write_ndarray(col_values.view('i8'), mask_to_file, &values)

        metadata.unit = TimeUnit_NANOSECOND

        tz = getattr(col.dtype, 'tz', None)
        if tz is None:
            metadata.timezone = b''
        else:
            metadata.timezone = tobytes(tz.zone)

        check_status(self.writer.get().AppendTimestamp(c_name, values,
                                                       metadata))

    cdef int write_ndarray(self, values, mask, PrimitiveArray* out) except -1:
        if mask is None:
            check_status(pandas_to_primitive(values, out))
        else:
            check_status(pandas_masked_to_primitive(values, mask, out))
        return 0


cdef _unbox_series(col):
    if isinstance(col, pd.Series):
        col_values = col.values
    else:
        col_values = col
    return col_values


cdef class Column:
    cdef:
        shared_ptr[CColumnMetadata] metadata
        CColumnMetadata* mp
        FeatherReader parent
        int column_index

    def __cinit__(self):
        self.mp = NULL

    cdef init(self, FeatherReader parent, int i):
        cdef TableReader* tbl = parent.reader.get()

        self.parent = parent
        self.column_index = i

        check_status(tbl.GetColumnMetadata(i, &self.metadata))
        self.mp = self.metadata.get()

    property name:

        def __get__(self):
            return frombytes(self.mp.name())

    property type:

        def __get__(self):
            return self.mp.type()

    property user_metadata:

        def __get__(self):
            return frombytes(self.mp.user_metadata())

    property null_count:
        def __get__(self):
            cdef:
                unique_ptr[CColumn] col
                CColumn* cp

            check_status(self.parent.reader.get()
                .GetColumn(self.column_index, &col))

            return col.get().values().null_count

    def read(self):
        cdef:
            unique_ptr[CColumn] col
            CColumn* cp

        check_status(self.parent.reader.get()
                     .GetColumn(self.column_index, &col))

        cp = col.get()

        if cp.type() == ColumnType_PRIMITIVE:
            values = primitive_to_pandas(cp.values())
        elif cp.type() == ColumnType_CATEGORY:
            values = category_to_pandas(cp)
        elif cp.type() == ColumnType_TIMESTAMP:
            values = timestamp_to_pandas(cp)
        else:
            raise NotImplementedError(cp.type())

        return values


cdef class FeatherReader:
    cdef:
        unique_ptr[TableReader] reader

    def __cinit__(self, object name):
        cdef:
            string c_name = encode_file_path(name)

        check_status(TableReader.OpenFile(c_name, &self.reader))

    property num_rows:

        def __get__(self):
            return self.reader.get().num_rows()

    property num_columns:

        def __get__(self):
            return self.reader.get().num_columns()

    def get_column(self, int i):
        if i < 0 or i >= self.num_columns:
            raise IndexError(i)

        cdef Column col = Column()
        col.init(self, i)

        return col


cdef category_to_pandas(CColumn* col):
    cdef CategoryColumn* cat = <CategoryColumn*>(col)

    values = raw_primitive_to_pandas(cat.values())
    mask = get_null_mask(cat.values())
    values[mask] = -1
    levels = primitive_to_pandas(cat.levels())

    return pd.Categorical(values, categories=levels,
                          fastpath=True)

cdef timestamp_to_pandas(CColumn* col):
    cdef TimestampColumn* ts = <TimestampColumn*>(col)

    values = raw_primitive_to_pandas(ts.values())
    mask = get_null_mask(ts.values())

    tz = frombytes(ts.timezone())
    if tz:
        values = (pd.DatetimeIndex(values).tz_localize('utc')
                  .tz_convert(tz))
        result = pd.Series(values)
    else:
        result = pd.Series(values, dtype='M8[ns]')
    result.iloc[mask] = pd.NaT

    return result
