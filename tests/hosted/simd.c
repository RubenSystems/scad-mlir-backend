
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "c_glue.h"

#define SIZE 1024


int32_t add(SCAD_ARRAY_DEF(int32_t), SCAD_ARRAY_DEF(int32_t), SCAD_ARRAY_DEF(int32_t));


__attribute__((noinline))
int32_t cadd (int32_t * a, int32_t * b, int32_t * c) {
	for (size_t jdx = 0; jdx < 1000000000; jdx ++) {
		// Equivilent to scad add
		for (size_t i = 0; i < SIZE; i ++) {
			c[i] = a[i] + b[i];
			
		}
	}
	return 0;
}


int32_t main() {
	int32_t * a = (int32_t *)malloc(SIZE * sizeof(int32_t));
	int32_t * b = (int32_t *)malloc(SIZE * sizeof(int32_t));
	int32_t * c = (int32_t *)malloc(SIZE * sizeof(int32_t));
	for (size_t i = 0; i < SIZE; i ++) {
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	


	int32_t add_res = add(SCAD_ARRAY(a, SIZE), SCAD_ARRAY(b, SIZE), SCAD_ARRAY(c, SIZE));
	// VS
	// 315 ms
	// cadd(a, b, c);
// 

	for (int i = 0; i < SIZE; i ++) {
		printf("%i\n", c[i]);
	}
	
}



