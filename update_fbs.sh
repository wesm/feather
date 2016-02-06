#!/bin/bash

FLATC=$HOME/local/bin/flatc
$FLATC --cpp -o src/feather src/feather/metadata.fbs
