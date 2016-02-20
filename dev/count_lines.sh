#!/usr/bin/env bash
cloc --by-file cpp/src/ --not-match-f="-test|generated|CMake"
