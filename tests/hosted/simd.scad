

fn tile_op(offset: ii, a: 1024xi32, b: 1024xi32, result: 1024xi32) i32 {

	let veca: 64xi32 = @vec.load(vec: a, offset: offset, size: 64_ii);
	let vecb: 64xi32 = @vec.load(vec: b, offset: offset, size: 64_ii);


	let resvec: 64xi32 = @add.v(a: veca, b: vecb);
    @vec.store(res: result, idx: resvec, offset: offset);

	0_i32
};

fn simd_add(a: 1024xi32, b: 1024xi32, result: 1024xi32) i32 {


	for i: 0_ii -> 16_ii {
		let offset: ii = @mul(a: i, b: 64_ii);
		tile_op(offset: offset, a: a, b: b, res: result);
	};
	0_i32
};


fn add(a: 1024xi32, b: 1024xi32, result: 1024xi32) i32 {
	
	for i: 0_ii -> 1000000000_ii {
		simd_add(a: a, b: b, res: result);
	};
	

	0_i32
};


