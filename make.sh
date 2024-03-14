#!/bin/bash

cd scad-lang
cargo build --lib
cd ../build
cmake --build .