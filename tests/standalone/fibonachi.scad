fn main() i32 {

	let sto = {1_i32, 1_i32}; 
	let end_res = 19_i32; 

	for i: 0_ii -> (@sub(a: end_res, b: 2_i32) -> ii) {
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