
To make: 

```sh
cmake -G Ninja $LLVM_SRC_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=scad \
    -DLLVM_EXTERNAL_SCAD_SOURCE_DIR=../

cmake --build .
```

