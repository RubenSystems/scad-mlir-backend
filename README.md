# SCAD MLIR Backend 

SCaD is a programming language for speed and efficiency. 

This is an entire compiler implementation so it can compiler your source code to machine code. 

Examples of scad: 

```
fn main() i32 {
	
	@print(value: @add(a: 100_i32, b: 200_i32));
	0_i32
};
```

```
fn main() i32 {

	for i: 0->100 step 2 {
		@print(v: i -> i32);
	};

	0_i32
};
```

if you want, you can unroll a loop for optimisation purposes: 
```
fn main() i32 {

	for i: 0->100 unroll 10 {
		@print(v: i -> i32);
	};

	0_i32
};
```

SCaD also supports multithreading using its parallel for:
```
fn main() i32 {

	parallel i: 0->100 {
		@print(v: i -> i32);
	};

	0_i32
};
```
There are not synchronisation primitaves yet. 

This is a more advanced program that adds two arrays. It takes advantage of SCaD's vector operations, which can map to SIMD instructions on hardware. See the `@vec.load`, `@add.v` and `@vec.store` intrinsics
```
fn tile_op(offset: ii, a: 128xi32, b: 128xi32, result: 128xi32) i32 {

	let veca: 16xi32 = @vec.load(vec: a, offset: offset, size: 16_ii);
	let vecb: 16xi32 = @vec.load(vec: b, offset: offset, size: 16_ii);
	let resvec: 16xi32 = @add.v(a: veca, b: vecb);
    @vec.store(res: result, idx: resvec, offset: offset);

	0_i32
};

fn simd_add(a: 128xi32, b: 128xi32, result: 128xi32) i32 {
	for i: 0 -> 8 {
		let offset: ii = @mul(a: i, b: 16_ii);
		tile_op(offset: offset, a: a, b: b, res: result);
	};
	0_i32
};


fn add(a: 128xi32, b: 128xi32, result: 128xi32) i32 {
	
	for i: 0 -> 1000000000 {
		simd_add(a: a, b: b, res: result);
	};
	

	0_i32
};
```

SCaD uses a hindly milner typing algorithm which allows for powerful type inferencing. In the below example, d is of type i32 (size_t or isize) 
```
let a = 0_ii;
let b = 0_ii;
let c = @add(a: a, b: b) -> i32;
let d = @add(c: c, d: 100_i32);
```

SCaD also has control flow. Conditionals are considered expressions, so the last expression evaluated is the value of the statement: 
```
let something = if @eq(a: 1_ii, b: 2_ii) {
    @add(a: 200, b: 400)
} else {
    @add(a: 200, b: 800);
}
```

To make, clone the repo (--recursive) and do this:

```sh
cmake -G Ninja $LLVM_SRC_DIR \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=scad \
    -DLLVM_EXTERNAL_SCAD_SOURCE_DIR=../

cmake --build .
```
