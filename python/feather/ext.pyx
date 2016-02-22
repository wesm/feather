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

from feather.compat import frombytes, tobytes
import six


class FeatherError(Exception):
    pass


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

    def close(self):
        pass

    cdef read_pandas_array(self, int i):
        pass

    cdef write_pandas_array(self, object name, object col):
        pass


cdef class FeatherReader:
    cdef:
        unique_ptr[TableReader] reader

    def __cinit__(self, object name):
        cdef:
            string c_name = tobytes(name)

        check_status(TableReader.OpenFile(c_name, &self.reader))


def write_pandas(df, path):
    '''
    Write a pandas.DataFrame to Feather format
    '''
    cdef FeatherWriter writer = FeatherWriter(path)

    # TODO(wesm): pipeline conversion to Arrow memory layout
    for name in df.columns:
        col = df[name]
        write_column(writer, name, col)

    writer.close()


def read_pandas(path, columns=None):
    """
    Read a pandas.DataFrame from Feather format

    Returns
    -------
    df : pandas.DataFrame
    """
    cdef:
        FeatherReader reader = FeatherReader(path)
        int i

    import pandas as pd

    # TODO(wesm): pipeline conversion to Arrow memory layout
    data = {}
    for i in range(reader.num_columns):
        name, arr = writer.read_pandas_array(i)
        data[name] = arr

    # TODO(wesm):
    return pd.DataFrame(data)
