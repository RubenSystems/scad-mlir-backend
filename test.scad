    fn main() 2xi32;

	fn super_add(a: 2xi32) 2xi32 {
		@add(a: @{1000, 1000}, b: a)
	};

    fn main() 2xi32 {
        let mut x: 2xi32 = @{700, 800};


        let mut super_x: 2xi32 = super_add(a: x);

        @print(val: super_x);

        let mut does_it_work: 2xi32 = if true {
            super_x
        } else {
            x
        };

        @print(value: does_it_work);

        does_it_work
    };