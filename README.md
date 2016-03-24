## Feather: fast, interoperable data frame storage for Python and R

Feather is a binary columnar serialization for data frames. It is designed to
read and write data frames as efficiently as possible, and is designed to work
across multiple languages. The initial version of feather comes with two
bindings, for python and for R.

Feather uses the [Apache Arrow](https://arrow.apache.org) columnar memory
specification for representing binary data on disk in a way that can be read
and written very fast. This is particularly important for null (NA) value
encoding as well as variable-length types like UTF8 strings. Note that Apache
Arrow does not provide a file format, and Feather defines its own schemas and
metadata to describe the structure of a Feather file.

Feather currently supports the following column types:

* A wide range of numeric types (int8, int16, int32, int64, uint8, uint16,
  uint32, uint64, float, double).
* Logical/boolean values.
* UTF-8 encoded strings.
* Dates, times, and timestamps.
* Factors/categorical variables with fixed set of possible values.
* Arbitrary binary data.

All column types support missing/null values.

## License and Copyrights

This library is released under the Apache License, Version 2.0.

See `NOTICE` for details about the library's copyright holders.
