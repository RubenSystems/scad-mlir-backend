
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define SCAD_ARRAY(pointer, size) pointer, pointer, 0, size, 1
#define SCAD_ARRAY_DEF(type) type *, type *, int64_t, int64_t, int64_t
#define SIZE 16

// extern "C" {
int32_t add(SCAD_ARRAY_DEF(int32_t), SCAD_ARRAY_DEF(int32_t), SCAD_ARRAY_DEF(int32_t));
// }


// __attribute__((noinline))
int32_t cadd (int32_t * a, int32_t * b, int32_t * c) {
	for (size_t jdx = 0; jdx < 100000000; jdx ++) {
		// Equivilent to scad add
		for (size_t i = 0; i < SIZE; i ++) {
			c[i] = a[i] + c[i];
			
		}
	}
	return 0;
}


int32_t main() {
	int32_t * a = (int32_t *)malloc(SIZE * sizeof(uint32_t));
	int32_t * b = (int32_t *)malloc(SIZE * sizeof(uint32_t));
	int32_t * c = (int32_t *)malloc(SIZE * sizeof(uint32_t));
	for (size_t i = 0; i < SIZE; i ++) {
		a[i] = 1;
		b[i] = 2;
		c[i] = 1;
	}


	int32_t add_res = add(SCAD_ARRAY(a, SIZE), SCAD_ARRAY(b, SIZE), SCAD_ARRAY(c, SIZE));
	// VS
	// cadd(a, b, c);
// 

	for (int i = 0; i < SIZE; i ++) {
		printf("%i\n", c[SIZE - 1]);
	}
	
}



