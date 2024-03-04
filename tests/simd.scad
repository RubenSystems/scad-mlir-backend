

fn tile_op(offset: ii, a: 64xi32, b: 64xi32, result: 64xi32) i32 {
	let mut veca: 16xi32 = @vec.load(vec: a, offset: offset, size: 16_ii);
	let mut vecb: 16xi32 = @vec.load(vec: b, offset: offset, size: 16_ii);
	let mut resvec: 16xi32 = @add.v(a: veca, b: vecb);
    @vec.store(res: result, idx: resvec, offset: 0_ii);

	0_i32
};

fn simd_add(a: 64xi32, b: 64xi32, result: 64xi32) i32 {
	for i : 0 -> 4 {
		let mut offset: ii = @mul(a: i, b: 16_ii);
		tile_op(offset: offset, a: a, b: b, res: result);
	};
	0_i32
};


fn add(a: 64xi32, b: 64xi32, result: 64xi32) i32 {
	
	simd_add(a: a, b: b, res: result);

	0_i32
};


