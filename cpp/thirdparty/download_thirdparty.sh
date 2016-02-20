#!/bin/bash

set -x
set -e

TP_DIR=$(cd "$(dirname "$BASH_SOURCE")"; pwd)

source $TP_DIR/versions.sh

download_extract_and_cleanup() {
	filename=$TP_DIR/$(basename "$1")
	curl -#LC - "$1" -o $filename
	tar xzf $filename -C $TP_DIR
	rm $filename
}

if [ ! -d ${GTEST_BASEDIR} ]; then
  echo "Fetching gtest"
  download_extract_and_cleanup $GTEST_URL
fi

if [ ! -d ${FLATBUFFERS_BASEDIR} ]; then
  echo "Fetching flatbuffers"
  download_extract_and_cleanup $FLATBUFFERS_URL
fi
