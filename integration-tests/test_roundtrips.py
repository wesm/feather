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

from pandas.util.testing import assert_frame_equal
import pandas as pd

import feather
import util


def test_factor_rep():
    fpath1 = util.random_path()
    fpath2 = util.random_path()

    rcode = """
library(feather)

iris <- read_feather("{0}")
iris$Species <- as.factor(as.character(iris$Species))
write_feather(iris, "{1}")
""".format(fpath1, fpath2)
    tmp_paths = []

    try:
        iris = pd.read_csv('iris.csv')
        levels = ['setosa', 'versicolor', 'virginica']

        iris['Species'] = pd.Categorical(iris['Species'], categories=levels)

        feather.write_dataframe(iris, fpath1)
        util.run_rcode(rcode)

        result = feather.read_dataframe(fpath2)

        tmp_paths.extend([fpath1, fpath2])
        assert_frame_equal(result, iris)
    finally:
        util.remove_paths(tmp_paths)
