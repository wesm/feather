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
