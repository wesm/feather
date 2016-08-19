# feather 0.3.0

* Added a `NEWS.md` file to track changes to the package.

* Time fields are now imported as class hms/difftime (#119).

* Timestamp (POSIXct) fields now correctly restore missing values (#157).

* UTF-8 field names are correctly imported (#198)

* Fixed protection bugs that lead to unpredictable crashes (#150, #204).

* Automatically close open file handles making it possible to read in
  hundreds of feather files in the same session (@krlmlr, #178)

* Also includes improvements to the underlying feather library

    * Support for > 2 Gb files on 32-bit windows.
    * Now works with earlier version of C++.


