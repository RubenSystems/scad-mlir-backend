fn idx(row: ii, col: ii) ii {
	@add(a: @mul(a: row, b: 1024_ii), b: col)
};

fn idx_32(row: i32, col: i32) i32 {
	@add(a: @mul(a: row , b: 1024_i32), b: col )
};

fn get(container: 1048576xi32, row: ii, column: ii) i32 {
	let gtidx = idx(r: row, c: column);
	@index.i32(c: container, idx: gtidx)
};

fn dot(a: 1048576xi32, b: 1048576xi32, result: 1048576xi32) i32 {


	for i: 0_ii -> 1024_ii {
		for j: 0_ii -> 1024_ii {
			for k: 0_ii -> 1024_ii {
				let a_get = get(container: a, r: i, c: k);
				let b_get = get(container: b, r: k, c: j);


				let res = @mul(a: a_get, b: b_get);
				let existing = @index.i32(container: result, idx: idx(r: i, c: j));
				let new = @add(a: res, b: existing);



				@set.i32(container: result, index: idx(r: i, c: j), value: new);
			};
		};
	};


	0_i32
};
