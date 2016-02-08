#!/usr/bin/env bash
cloc --by-file src/ --not-match-f="-test|generated|CMake"
