#pragma once

#define SCAD_ARRAY(pointer, size) pointer, pointer, 0, size, 1
#define SCAD_ARRAY_DEF(type) type *, type *, int64_t, int64_t, int64_t 