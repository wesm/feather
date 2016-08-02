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
import sys

import numpy as np

from pandas.util.testing import assert_frame_equal
import pandas as pd

import feather

import uuid

nrows = 4000000
ncols = 100

data = np.random.randn(nrows)

df = pd.DataFrame({'c{0}'.format(i): data
                   for i in range(ncols)})

def guid():
    return uuid.uuid4().hex

path = 'test_{0}.feather'.format(guid())

try:
    feather.write_dataframe(df, path)
    df2 = feather.read_dataframe(path)
    assert_frame_equal(df, df2)
finally:
    try:
        os.remove(path)
    except os.error:
        pass
