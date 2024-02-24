


fn add(a: 10000xi32, b: 10000xi32, result: 10000xi32) i32 {
	for i: 0 -> 10000 {
		let mut a_idx_at: i32 = @index.i32(container: a, index: i);
		let mut b_idx_at: i32 = @index.i32(container: b, index: i);
		let mut addres: i32 = @add(a: a_idx_at, b: b_idx_at);
		@set.i32(r: result, i: i, v: addres);
	};

	0
};