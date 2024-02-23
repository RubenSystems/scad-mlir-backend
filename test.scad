
fn do(b: i32, a: i32) i32 {

	@index.i32(a: @{1,2,3}, b: b)
};

fn main() i32 {
	let mut x: i32 = 100;

	let mut doing: i32 = do(a: x, b: x);
	let mut something_rlly_cl: i32 = @add(a: 1, b: 8);

	@print(a: something_rlly_cl);



	doing
};
