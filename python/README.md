## Python interface to the Apache Arrow-based Feather File Format

Feather efficiently stores pandas DataFrame objects on disk.

## Build

Building Feather requires a C++11 compiler. We've simplified the PyPI packaging
to include libfeather (the C++ core library) to be built statically as part of
the Python extension build, but this may change in the future.

### Static builds for easier packaging

At the moment, the libfeather sources are being built and linked with the
Cython extension, rather than building the `libfeather` shared library and
linking to that.

While we continue to do this, building from source requires you to symlink (or
copy) the C++ sources. See:

```shell
# Symlink the C++ library for the static build
ln -s ../cpp/src src
python setup.py build

# To install it locally
python setup.py install

# Source distribution
python setup.py sdist
```

To change this and instead link to an installed `libfeather.so`, look in
`setup.py` and make the following change:

```python
FEATHER_STATIC_BUILD = False
```

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
