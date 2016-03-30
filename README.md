## Feather: fast, interoperable data frame storage

[![Travis-CI Build Status](https://travis-ci.org/wesm/feather.svg?branch=master)](https://travis-ci.org/wesm/feather) [![Coverage Status](https://img.shields.io/codecov/c/github/wesm/feather/master.svg)](https://codecov.io/github/wesm/feather?branch=master)

Feather is binary columnar serialization for data frames. It is designed to
read and write data frames very efficiently, and to make it easy to share data
across multiple data analysis languages. The initial version of Feather comes
with bindings for [python](python/) (written by [Wes
McKinney](https://github.com/wesm)) and [R](R/) (written by [Hadley
Wickham](https://github.com/hadley/)).

Feather uses the [Apache Arrow](https://arrow.apache.org) columnar memory
specification to representing binary data on disk in a way that can be read
and written very rapidly. This is particularly important for encoding
null/NA values and variable-length types like UTF8 strings. Feather is
complementary to Apache Arrow because Arrow does not provide a file format,
so Feather defines its own schemas and metadata for an on-disk representation.

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
