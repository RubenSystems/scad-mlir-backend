

fn main() i32 {
	let number = {1_i32};

	for i: 1_ii -> 10_ii {
		let idx_val = @index.i32(value: number, idx: 0_ii);
		let res = @mul(a: idx_val, b: i -> i32);
		@set.i32(container: number, index: 0_ii, value: res);
		@print(val: res);
	};

	@print(val: @index.i32(value: number, idx: 0_ii));

	0_i32
};