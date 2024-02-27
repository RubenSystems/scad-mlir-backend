#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <argument> <arguement>"
    exit 1
fi

echo "Compiling scad program $1"
build/bin/scadc $1
llc -filetype=obj -o $3 $2

