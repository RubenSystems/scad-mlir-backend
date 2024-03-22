fn main() i32 {
	
	
	let a = 1_i32; 
	let b = @add(a: a, b: 1_i32);
	let c = @add(a: b, b: 1_i32);


	@print(value: c);


	0_i32
};

