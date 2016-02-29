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

import os
import unittest

import numpy as np

from pandas.util.testing import assert_frame_equal
import pandas as pd

from feather.compat import guid
from feather import FeatherReader, FeatherWriter
import feather


def random_path():
    return 'feather_{}'.format(guid())


class TestFeatherReader(unittest.TestCase):

    def setUp(self):
        self.test_files = []

    def tearDown(self):
        for path in self.test_files:
            try:
                os.remove(path)
            except os.error:
                pass

    def test_file_not_exist(self):
        with self.assertRaises(feather.FeatherError):
            FeatherReader('test_invalid_file')

    def _check_pandas_roundtrip(self, df):
        path = random_path()
        self.test_files.append(path)
        feather.write_dataframe(df, path)
        if not os.path.exists(path):
            raise Exception('file not written')
        result = feather.read_dataframe(path)
        assert_frame_equal(result, df)

    def test_float_no_nulls(self):
        data = {}
        numpy_dtypes = ['f4', 'f8']
        num_values = 100

        for dtype in numpy_dtypes:
            values = np.random.randn(num_values)
            data[dtype] = values.astype(dtype)

        df = pd.DataFrame(data)
        self._check_pandas_roundtrip(df)

    def test_float_nulls(self):
        num_values = 100

        path = random_path()
        self.test_files.append(path)
        writer = FeatherWriter(path)

        null_mask = np.random.randint(0, 10, size=num_values) < 3
        dtypes = ['f4', 'f8']
        expected_cols = []
        for name in dtypes:
            values = np.random.randn(num_values).astype(name)
            writer.write_array(name, values, null_mask)

            values[null_mask] = np.nan

            expected_cols.append(values)

        writer.close()

        ex_frame = pd.DataFrame(dict(zip(dtypes, expected_cols)),
                                columns=dtypes)

        result = feather.read_dataframe(path)
        assert_frame_equal(result, ex_frame)

    def test_integer_no_nulls(self):
        data = {}

        numpy_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
        num_values = 100

        for dtype in numpy_dtypes:
            info = np.iinfo(dtype)
            values = np.random.randint(info.min,
                                       min(info.max, np.iinfo('i8').max),
                                       size=num_values)
            data[dtype] = values.astype(dtype)

        df = pd.DataFrame(data)
        self._check_pandas_roundtrip(df)

    def test_integer_with_nulls(self):
        # pandas requires upcast to float dtype
        path = random_path()
        self.test_files.append(path)

        int_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
        num_values = 100

        writer = FeatherWriter(path)

        null_mask = np.random.randint(0, 10, size=num_values) < 3
        expected_cols = []
        for name in int_dtypes:
            values = np.random.randint(0, 100, size=num_values)
            writer.write_array(name, values, null_mask)

            expected = values.astype('f8')
            expected[null_mask] = np.nan

            expected_cols.append(expected)

        ex_frame = pd.DataFrame(dict(zip(int_dtypes, expected_cols)),
                                columns=int_dtypes)

        writer.close()

        result = feather.read_dataframe(path)
        assert_frame_equal(result, ex_frame)

    def test_boolean_no_nulls(self):
        num_values = 100

        np.random.seed(0)

        df = pd.DataFrame({'bools': np.random.randn(num_values) > 0})
        self._check_pandas_roundtrip(df)

    def test_boolean_nulls(self):
        # pandas requires upcast to object dtype
        pass

    def test_category(self):
        pass
