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

import six
from distutils.version import LooseVersion
import pandas as pd

import feather.ext as ext

if LooseVersion(pd.__version__) < '0.17.0':
    raise ImportError("feather requires pandas >= 0.17.0")


def write_dataframe(df, path):
    '''
    Write a pandas.DataFrame to Feather format
    '''
    writer = ext.FeatherWriter(path)
    
    if isinstance(df, pd.sparse.frame.SparseDataFrame):
        df = df.to_dense()

    # TODO(wesm): pipeline conversion to Arrow memory layout
    for i, name in enumerate(df.columns):
        col = df.iloc[:, i]

        if not isinstance(name, six.string_types):
            name = str(name)

        writer.write_array(name, col)

    writer.close()


def read_dataframe(path, columns=None):
    """
    Read a pandas.DataFrame from Feather format

    Parameters
    ----------
    path : string, path to read from
    columns : sequence, optional
        Only read a specific set of columns. If not provided, all columns are
        read

    Returns
    -------
    df : pandas.DataFrame
    """
    reader = ext.FeatherReader(path)

    if columns is not None:
        columns = set(columns)

    # TODO(wesm): pipeline conversion to Arrow memory layout
    data = {}
    names = []
    for i in range(reader.num_columns):
        col = reader.get_column(i)
        name = col.name
        if columns is None or name in columns:
            arr = col.read()
            data[name] = arr
            names.append(name)

    # TODO(wesm):
    return pd.DataFrame(data, columns=names)
