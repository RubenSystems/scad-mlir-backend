fn main() i32 {

	let sto = {1_i32, 1_i32}; 

	for i: 0_ii->17_ii {
		let a = @index.i32(c: sto, idx: 0_ii);
		let b = @index.i32(c: sto, idx: 1_ii);

		let c = @add(a: a, b: b);

		@set.i32(c: sto, idx: 0_ii, value: b);
		@set.i32(c: sto, idx: 1_ii, value: c);

	};

	let end = @index.i32(c: sto, idx: 1_ii);
	@print(v: end);

	0_i32
};
