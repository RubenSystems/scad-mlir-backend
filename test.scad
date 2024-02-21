fn main() 2xi32;

fn add_200(a: 2xi32, b: 2xi32) 2xi32 {
	@add(a: a, b: @add(a: b, b: @{200, 200}))
};

fn main() 2xi32 {
	let mut x: 2xi32 = @{700, 800};

	@print(value: x);

	x
};
