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
import pandas.core.common as pdcom

from numpy cimport ndarray
cimport numpy as cnp
import numpy as np

from feather.compat import frombytes, tobytes
import six

cnp.import_array()

class FeatherError(Exception):
    pass

cdef extern from "interop.h" namespace "feather::py":
    Status pandas_to_primitive(object ao, PrimitiveArray* out)
    Status pandas_masked_to_primitive(object ao, object mask,
                                      PrimitiveArray* out)
    object primitive_to_pandas(const PrimitiveArray& arr)

cdef check_status(const Status& status):
    if status.ok():
        return

    cdef string c_message = status.ToString()
    raise FeatherError(frombytes(c_message))

cdef object numpy_nan = np.nan

cdef class FeatherWriter:
    cdef:
        unique_ptr[TableWriter] writer
        int64_t num_rows

    def __cinit__(self, object name):
        cdef:
            string c_name = tobytes(name)

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

        if pdcom.is_categorical_dtype(col.dtype):
            raise NotImplementedError(col.dtype)
        else:
            self.write_primitive(name, col, mask)

    cdef write_primitive(self, name, col, mask):
        cdef:
            string c_name = tobytes(name)
            PrimitiveArray values

        if isinstance(col, pd.Series):
            col_values = col.values
        else:
            col_values = col

        if mask is None:
            check_status(pandas_to_primitive(col_values, &values))
        else:
            check_status(pandas_masked_to_primitive(
                col_values, mask, &values))

        check_status(self.writer.get().AppendPlain(c_name, values))


cdef class FeatherReader:
    cdef:
        unique_ptr[TableReader] reader

    def __cinit__(self, object name):
        cdef:
            string c_name = tobytes(name)

        check_status(TableReader.OpenFile(c_name, &self.reader))

    property num_rows:

        def __get__(self):
            return self.reader.get().num_rows()

    property num_columns:

        def __get__(self):
            return self.reader.get().num_columns()

    def read_array(self, int i):
        cdef:
            shared_ptr[Column] col
            Column* cp
            CategoryColumn* cat

        if i < 0 or i >= self.num_columns:
            raise IndexError(i)

        check_status(self.reader.get().GetColumn(i, &col))

        cp = col.get()

        if cp.type() == ColumnType_PRIMITIVE:
            values = primitive_to_pandas(cp.values())
        elif cp.type() == ColumnType_CATEGORY:
            values = category_to_pandas(cp)
        else:
            raise NotImplementedError(cp.type())

        return frombytes(cp.name()), values


cdef category_to_pandas(Column* col):
    cdef CategoryColumn* cat = <CategoryColumn*>(col)

    values = primitive_to_pandas(cat.values())
    levels = primitive_to_pandas(cat.levels())

    return pd.Categorical(values=values, categories=levels)
