fn idx(row: ii, col: ii) ii {
	@add(a: @mul(a: row, b: 2_ii), b: col)
};

fn get(container: 4xi32, row: ii, column: ii) i32 {
	let gtidx = idx(r: row, c: column);
	@index.i32(c: container, idx: gtidx)
};

fn dot(a: 4xi32, b: 4xi32) 4xi32 {
	let result = @empty(filler: 0_i32, size: 4_ii);

	for i: 0_ii -> 2_ii {
		for j: 0_ii -> 2_ii {
			for k: 0_ii -> 2_ii {
				let res = @mul(a: get(container: a, r: i, c: k), b: get(container: b, r: k, c: j));
				let existing = @index.i32(container: result, idx: idx(r: i, c: j));

				@set.i32(container: result, index: idx(r: i, c: j), value: @add(a: res, b: existing));
			};
		};
	};


	result
};

fn main() i32 {

	let a = {1_i32, 2_i32, 3_i32, 4_i32};
	let b = {1_i32, 2_i32, 3_i32, 4_i32};

	let dot_val = dot(a: a, b: b);

	for i: 0_ii -> 4_ii {
		@print(a: @index.i32(container: dot_val, index: i));
	};


	0_i32
};