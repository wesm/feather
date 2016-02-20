SOURCE_DIR=$(cd "$(dirname "$BASH_SOURCE")"; pwd)

source thirdparty/versions.sh

export FLATBUFFERS_HOME=$SOURCE_DIR/thirdparty/installed
export GTEST_HOME=$SOURCE_DIR/thirdparty/$GTEST_BASEDIR
