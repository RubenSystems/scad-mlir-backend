

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


