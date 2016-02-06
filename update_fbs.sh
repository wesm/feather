#!/bin/bash

FLATC=$HOME/local/bin/flatc
$FLATC -c -o src/feather src/feather/metadata.fbs
