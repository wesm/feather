## R Feather Format API


## Bootstrapping

```sh
mkdir -p src/libfeather
cp ../cpp/src/feather/*.h src/feather
cp ../cpp/src/feather/*.cc src/feather
rm src/feather/*-test.cc src/feather/test_main.cc src/feather/test-common.h

mkdir -p src/flatbuffers
cp ../cpp/thirdparty/flatbuffers-1.3.0/include/flatbuffers/flatbuffers.h src/flatbuffers/
```

## Installation

Install from Github with:

```R
# install.packages("devtools")
devtools::install_github("wesm/feather/R")
```
