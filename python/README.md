## Python interface to the Apache Arrow-based Feather File Format

Feather efficiently stores pandas DataFrame objects on disk.

## Build

Building Feather requires a C++11 compiler. We've simplified the PyPI packaging
to include libfeather (the C++ core library) to be built statically as part of
the Python extension build, but this may change in the future.

## Limitations

Some features of pandas are not supported in Feather:

* Non-string column names
* Row indexes
* Object-type columns with non-homogeneous data

## Mac notes

Anaconda uses a default 10.5 deployment target which does not have C++11
properly available. This can be fixed by setting:

```
export MACOSX_DEPLOYMENT_TARGET=10.10
```
