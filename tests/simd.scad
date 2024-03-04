

fn tile_add(a: 16xi32, b: 16xi32, result: 16xi32) i32 {

	let mut veca: 16xi32 = @vec.load(vec: a, offset: 0_ii, size: 16_ii);
	let mut vecb: 16xi32 = @vec.load(vec: b, offset: 0_ii, size: 16_ii);
	let mut resvec: 16xi32 = @add.v(a: veca, b: vecb);
    @vec.store(res: result, idx: resvec, val: 0_ii);

	320_i32
};

