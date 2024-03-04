fn add(a: 32xi32, b: 32xi32, result: 32xi32) i32 {
	for i: 0 -> 32 {
		@add.v(a: a, b: b, res: result);
	};

	320_i32
};

