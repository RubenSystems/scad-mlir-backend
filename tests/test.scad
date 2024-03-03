fn add(a: 100000xi32, b: 100000xi32, result: 100000xi32) i32 {
	for i: 0 -> 100000 {
		for j: 0 -> 100000 {
			let mut a_b: i32 = @add(a: @index.i32(c: a, idx: j), b: @index.i32(c: b, idx: j));
			@set.i32(container: result, idx: j, res: a_b);
		};
		let mut current_c: i32 = @index.i32(c: result, idx: i);
		let mut current_a: i32 = @index.i32(c: a, idx: i);
		let mut current_b: i32 = @index.i32(c: b, idx: i);
		let mut add_ab: i32 = @add(a: current_a, b: current_b);
		let mut total: i32 = @add(a: current_c, b: add_ab);
		@set.i32(container: result, index: i, res: total);
	};

	1000000_i32
};

