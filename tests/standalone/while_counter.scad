fn main() i32 {

	let value = @{0_i32};
	while @lt(a: @index.i32(c: value, idx: 0_ii), b: 100_i32) {
		let new_val: i32 = @add(a: @index.i32(c: value, idx: 0_ii), b: 1_i32);
		@set.i32(c: value, i: 0_ii, v: new_val);
		@print(v: @index.i32(c: value, idx: 0_ii));
	};


	0_i32
};
