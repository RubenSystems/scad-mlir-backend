# SCAD MLIR Backend 

MLIR Dialect for scad. 

Will support ScadTLCIR -> SCAD MLIR Dialect -> LLVM IR lowering .


To make: 

```sh
cmake -G Ninja $LLVM_SRC_DIR \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=scad \
    -DLLVM_EXTERNAL_SCAD_SOURCE_DIR=../

cmake --build .
```

```sh
llc -filetype=obj -o output.o output.ll
```