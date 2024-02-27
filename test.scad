fn add(a: 10000xi64, b: 10000xi64, result: 10000xi64) i64 {
	let mut thing: 1xi32 = @{100_i32};
	for i: 0 -> 10000 {
		let mut a_v: i32 = 100_i32;
		let mut b_v: i32 = 200_i32;
		let mut addv: i32 = @add(a: a_v, b: b_v);
		@set.i32(a: thing, b: 0_ii, c: addv);
	};

	10000_i64
};