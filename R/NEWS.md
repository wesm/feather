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
