fn do(a: 2xi32, b: 2xi32) 2xi32 {
	@{300, 400}
};


fn main() i32 {
	let mut x: i32 = 100;
	
	let mut container: 2xi32 = @{x, x};


	let mut y: 2xi32 = do(a: container, b: container);

	x
};
