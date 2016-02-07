#!/usr/bin/env bash

# Build an isolated thirdparty
cp -r $TRAVIS_BUILD_DIR/thirdparty .
./thirdparty/download_thirdparty.sh
source thirdparty/versions.sh

# this is created in Travis CI
export BUILD_DIR=$HOME/build_dir
export TP_DIR=$BUILD_DIR/thirdparty

if [ $TRAVIS_OS_NAME == "osx" ]; then
  ./thirdparty/build_thirdparty.sh
fi

if [ $TRAVIS_OS_NAME == "linux" ]; then
  ./thirdparty/build_thirdparty.sh
  # Use a C++11 compiler on Linux
  export CC="gcc-4.9"
  export CXX="g++-4.9"
fi

export GTEST_HOME=$TP_DIR/$GTEST_BASEDIR
export FLATBUFFERS_HOME=$TP_DIR/installed

FEATHER_SRC=$TRAVIS_BUILD_DIR/src/feather

FLATC=$TP_DIR/installed/bin/flatc
$FLATC -c -o $FEATHER_SRC $FEATHER_SRC/metadata.fbs
