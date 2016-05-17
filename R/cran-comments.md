This is a resubmission that adds missing authors information. Sorry for the omission!

---

## Test environments
* local OS X install, R 3.3.0
* ubuntu 12.04 (on travis-ci), R 3.3.0
* win-builder (devel and release)

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

We have tested on Solaris and it appears to work (despite a downstream dependency, flatbuffers, not explicitly documenting Solaris support). The feather R package will not work on big-endian platforms because the underlying library that we are wrapping (the feather C++ library), does not currently support big-endian platforms. Please let me know if there's some way to cleanly document this in the DESCRIPTION.

## Reverse dependencies

This is a new release, so there are no reverse dependencies.
