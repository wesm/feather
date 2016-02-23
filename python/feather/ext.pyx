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


cdef check_status(const Status& status):
    if status.ok():
        return

    cdef string c_message = status.ToString()
    raise FeatherError(frombytes(c_message))


cdef class FeatherWriter:
    cdef:
        unique_ptr[TableWriter] writer
        int64_t num_rows

    def __cinit__(self, object name):
        cdef:
            string c_name = tobytes(name)

        check_status(TableWriter.OpenFile(c_name, &self.writer))

    def close(self):
        check_status(self.writer.get().Finalize())

    cdef write_series(self, object name, object col):
        if pdcom.is_categorical_dtype(col.dtype):
            raise NotImplementedError(col.dtype)
        else:
            self.write_primitive(name, col)

    cdef write_primitive(self, name, col):
        cdef:
            string c_name = tobytes(name)
            PrimitiveArray values

        check_status(pandas_to_primitive(col.values, &values))
        check_status(self.writer.get().AppendPlain(name, values))


cdef class FeatherReader:
    cdef:
        unique_ptr[TableReader] reader

    def __cinit__(self, object name):
        cdef:
            string c_name = tobytes(name)

        check_status(TableReader.OpenFile(c_name, &self.reader))

    property num_columns:

        def __get__(self):
            return self.reader.get().num_columns()

    cdef read_series(self, int i):
        pass


def write_dataframe(df, path):
    '''
    Write a pandas.DataFrame to Feather format
    '''
    cdef FeatherWriter writer = FeatherWriter(path)

    # TODO(wesm): pipeline conversion to Arrow memory layout
    for name in df.columns:
        col = df[name]
        writer.write_series(name, col)

    writer.close()


def read_dataframe(path, columns=None):
    """
    Read a pandas.DataFrame from Feather format

    Returns
    -------
    df : pandas.DataFrame
    """
    cdef:
        int i
        FeatherReader reader = FeatherReader(path)

    # TODO(wesm): pipeline conversion to Arrow memory layout
    data = {}
    for i in range(reader.num_columns):
        name, arr = reader.read_series(i)
        data[name] = arr

    # TODO(wesm):
    return pd.DataFrame(data)
