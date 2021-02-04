## Feather Development is in Apache Arrow now

Feather development lives on in [Apache
Arrow](https://github.com/apache/arrow). The `arrow` R package includes [a much
faster implementation of
Feather](http://arrow.apache.org/blog/2019/08/08/r-package-on-cran/),
i.e. `arrow::read_feather`. The Python package `feather` is now a wrapper
around `pyarrow.feather`.

## Feather: fast, interoperable data frame storage

Feather provides binary columnar serialization for data frames. It is designed to
make reading and writing data frames efficient, and to make sharing data across
data analysis languages easy. This initial version comes with bindings for
[python](python/) (written by [Wes McKinney](https://github.com/wesm)) and [R](R/)
(written by [Hadley Wickham](https://github.com/hadley/)).

Feather uses the [Apache Arrow](https://arrow.apache.org) columnar memory
specification to represent binary data on disk. This makes read and write
operations very fast. This is particularly important for encoding null/NA values
and variable-length types like UTF8 strings.

Feather is a part of the broader Apache Arrow project. Feather defines its own
simplified schemas and metadata for on-disk representation.

Feather currently supports the following column types:

* A wide range of numeric types (int8, int16, int32, int64, uint8, uint16,
  uint32, uint64, float, double).
* Logical/boolean values.
* Dates, times, and timestamps.
* Factors/categorical variables that have fixed set of possible values.
* UTF-8 encoded strings.
* Arbitrary binary data.

All column types support NA/null values.

## Installation

### Python

`pip install feather-format`

### R

`install.packages("feather")`

## Julia

```
julia> using Pkg
julia> Pkg.add("Feather")
```

## License and Copyrights

This library is released under the [Apache License, Version 2.0](LICENSE.txt).

See [`NOTICE`](NOTICE) for details about the library's copyright holders.

## Getting started

### Python

To get started with the python bindings, see the [python feather documentation](https://arrow.apache.org/docs/python/feather.html)

### R

To get started with the R bindings, see the [R feather documentation](https://cran.r-project.org/web/packages/feather/feather.pdf)

### Julia

To get started with the Julia bindings see [Feather.jl](https://github.com/JuliaStats/Feather.jl)
