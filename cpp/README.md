## Feather core C++ library

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

## Build and install

`libfeather` is mainly designed for linking with some other application, like a
Python or R extension. To build the dependencies and install a release build,
run:

```shell
./thirdparty/download_thirdparty.sh
./thirdparty/build_thirdparty.sh

# Sets the thirdparty build environment variables
source develop_env.sh

# Generates the flatbuffers bindings, to be automated
./update_fbs.sh

mkdir release-build
cd release-build
export FEATHER_HOME=$HOME/local
cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX=$FEATHER_HOME ..
make -j4
make install
```

Now, make sure that `$FEATHER_HOME` is used for building with your thirdparty
application.

## Development info

The core libfeather library is implemented in C++11.

We use [Google C++ coding style][1] with a few changes:

- Long lines are permitted up to 90 characters
- We do not encourage anonymous namespaces

We do not use C++ exceptions as handling them in Python extensions adds a lot
of library complexity. Instead return a `Status` object. This also makes it
simpler to make libfeather accessible to other C users.

Style is checked with `cpplint`, after generating the make files you can
veryify the style with

```
make lint
```

Explicit memory management and non-trivial destructors are to be avoided. Use
smart pointers to manage the lifetime of objects and memory, and generally use
[RAII][2] whenever possible.

[1]: http://google.github.io/styleguide/cppguide.html
[2]: https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization
