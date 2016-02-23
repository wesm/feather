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

import feather.ext as feather


def guid():
    import uuid
    return uuid.uuid4().get_hex()


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
            feather.FeatherReader('test_invalid_file')

    def _check_pandas_roundtrip(self, df):
        path = guid()
        feather.write_dataframe(df, path)
        result = feather.read_dataframe(path)
        assert_frame_equal(result, df)

    def test_float_types(self):
        pass

    def test_integer_types(self):
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
        pass
