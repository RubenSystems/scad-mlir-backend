
#include "c_glue.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int32_t dot(SCAD_ARRAY_DEF(int32_t), SCAD_ARRAY_DEF(int32_t), SCAD_ARRAY_DEF(int32_t));

#define SIZE 1024 * 1024

int main() {
	
	int32_t * a = malloc(sizeof(int32_t) * SIZE);
	int32_t * b = malloc(sizeof(int32_t) * SIZE);

	for (int i = 0; i < SIZE; i ++) {
		a[i] = 1; 
		b[i] = 1; 
	}

	int32_t * result = malloc(sizeof(int32_t) * SIZE);
	memset(result, 0, sizeof(int32_t) * SIZE);

	dot(SCAD_ARRAY(a, SIZE), SCAD_ARRAY(b, SIZE), SCAD_ARRAY(result, SIZE));

	for (int i = 0; i < SIZE; i ++) {
		printf("%i ", result[i]);
	}
}