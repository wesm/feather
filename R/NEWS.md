# feather 0.4.0

This release updates `feather` to depend on the [`arrow`](https://arrow.apache.org/docs/r/) package, which is where Feather format development has moved. This resolves many bug reports and missing features from the previous implementation of Feather in R, and it brings the R package in line with the Python `feather` package, which has depended on `pyarrow` since 2017.

For compatibility, `feather::write_feather()` uses V1 of the Feather specification: it is a wrapper around `arrow::write_feather(version = 1)`. Feather V2 is just the Arrow format on disk and has support for a much richer set of data types. To switch to V2, we recommend just using `arrow::write_feather()`.

With these changes, most of `feather`'s APIs are preserved in spirit, but there are some noteworthy changes:

* The `feather` class, which allowed for accessing data in a Feather file without reading it all into R, has been replaced by an `arrow::Table` backed by the memory-mapped Feather file. This should preserve the intent of the original implementation but with much richer functionality. One behavior change that results though is that slicing/extracting from the Table results in another arrow Table, so the data aren't pulled into a `data.frame` until you `as.data.frame()` them.

* `feather_metadata` also now does the same.

# feather 0.3.5

* Fixes for CRAN

# feather 0.3.4

* Fixes for CRAN

# feather 0.3.3

* `feather_metadata()` now handles paths with `~` in them.

* Fix warnings on CRAN due to (unused) GNU Makefiles.

* Use `tibble::tibble()` in place of the deprecated `dplyr::data_frame()`.

# feather 0.3.2

* Fixes for PROTECT error found by rchk.

* Use native routine registration

* Fix test failure due to UTF-8 encoded paths on Windows.

# feather 0.3.1

## Bug fixes and minor features

* Fixes for gcc-4.4 compilation.

# feather 0.3.0

## Underlying feather library changes

* Feather files are now padded to 64-bit alignment. Feather 0.3 will
  read feather files created by the previous version, but this will not
  be true for future versions of feather.

* Support for > 2 Gb files on 32-bit windows.

* Now works with earlier version of C++.

## Bug fixes and minor features

* Automatically close open file handles making it possible to read in
  hundreds of feather files in the same session (@krlmlr, #178)

* Added a `NEWS.md` file to track changes to the package.

* Fixed protection bugs that lead to unpredictable crashes (#150, #204).

* Time fields are now imported as class hms/difftime (#119).

* Timestamp (POSIXct) fields now correctly restore missing values (#157).

* UTF-8 field names are correctly imported (#198)
