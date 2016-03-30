## Feather core C++ library

## Build and install

`libfeather` is mainly designed for linking with some other application, like a
Python or R extension. To build the dependencies and install a release build,
run:

```shell
./thirdparty/download_thirdparty.sh
./thirdparty/build_thirdparty.sh

# Sets the thirdparty build environment variables
source develop_env.sh

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
