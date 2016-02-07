#!/bin/bash

FLATC=thirdparty/installed/bin/flatc
$FLATC -c -o src/feather src/feather/metadata.fbs
