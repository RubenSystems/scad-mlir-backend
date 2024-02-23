
fn do(b: i32, a: i32) i32 {

	@index.i32(a: @{1,2,3}, b: b)
};

fn main() i32 {
	let mut x: i32 = 100;

	let mut doing: i32 = do(a: x, b: x);
	let mut something_rlly_cl: i32 = @add(a: 1, b: 8);

	let mut array_to_loop_over: 10xi32 = @{1,2,3,4,5,6,7,8,100,200};
	for i: 0 -> 10 {
		@print(val: @index.i32(arr: array_to_loop_over, idx: i));
	};
 
	doing
};


