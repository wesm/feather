## Feather: fast, interoperable data frame storage

Testing GerritHub

[![Travis-CI Build Status](https://travis-ci.org/wesm/feather.svg?branch=master)](https://travis-ci.org/wesm/feather) [![Coverage Status](https://img.shields.io/codecov/c/github/wesm/feather/master.svg)](https://codecov.io/github/wesm/feather?branch=master)
[![PyPI](https://img.shields.io/pypi/v/feather-format.svg?maxAge=3600)](https://pypi.python.org/pypi/feather-format)

Feather provides binary columnar serialization for data frames. It is designed to
make reading and writing data frames efficient, and to make sharing data across
data analysis languages easy. This initial version comes with bindings for
[python](python/) (written by [Wes McKinney](https://github.com/wesm)) and [R](R/)
(written by [Hadley Wickham](https://github.com/hadley/)).

Feather uses the [Apache Arrow](https://arrow.apache.org) columnar memory
specification to represent binary data on disk. This makes read and write
operations very fast. This is particularly important for encoding null/NA values
and variable-length types like UTF8 strings.

Feather is complementary to Apache Arrow. Because Arrow does not provide a file
format, Feather defines its own schemas and metadata for on-disk representation.

Feather currently supports the following column types:

* A wide range of numeric types (int8, int16, int32, int64, uint8, uint16,
  uint32, uint64, float, double).
* Logical/boolean values.
* Dates, times, and timestamps.
* Factors/categorical variables that have fixed set of possible values.
* UTF-8 encoded strings.
* Arbitrary binary data.

All column types support NA/null values.

## License and Copyrights

This library is released under the [Apache License, Version 2.0](LICENSE.txt).

See [`NOTICE`](NOTICE) for details about the library's copyright holders.
