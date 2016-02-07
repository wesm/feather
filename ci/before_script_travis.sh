#!/usr/bin/env bash

# Build an isolated thirdparty
cp -r $TRAVIS_BUILD_DIR/thirdparty .
./thirdparty/download_thirdparty.sh
source thirdparty/versions.sh

if [ $TRAVIS_OS_NAME == "osx" ]; then
  # Only build gtest
  ./thirdparty/build_thirdparty.sh gtest
fi

if [ $TRAVIS_OS_NAME == "linux" ]; then
  ./thirdparty/build_thirdparty.sh
  # Use a C++11 compiler on Linux
  export CC="gcc-4.9"
  export CXX="g++-4.9"
fi

export GTEST_HOME=$HOME/build_dir/thirdparty/$GTEST_BASEDIR
export FLATBUFFERS_HOME=$HOME/build_dir/thirdparty/installed

FLATC=$HOME/build_dir/thirdparty/installed/bin/flatc
$FLATC -c -o src/feather src/feather/metadata.fbs
