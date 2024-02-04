#pragma once

#include <stdint.h>
#include <string>
#include <stddef.h>

extern "C" {
	struct FFIHIRArray {
		const struct FFIHIRValue * vals;
		size_t size;
	};

	struct FFIHIRInteger {
		size_t value;
	};

	union ValueUnion {
		FFIHIRArray array;
		FFIHIRInteger integer;
	};

	enum FFIHirValueTag { Array, Integer };

	struct FFIHIRValue {
		FFIHirValueTag tag;
		ValueUnion value;
	};

	enum FFIHIRTag {
		VariableDecl = 0,
		Noop = 1,
		FunctionDecl = 2,
		ForwardFunctionDecl = 3,
		Return = 4,
	};

	struct FFIHIRVariableDecl {
		const char * name;
		struct FFIHIRValue e1;
		struct FFIHIRExpr * e2;
	};

	struct FFIHIRFunctionDecl {
		const char * name;
		const struct FFIHIRExpr * block;
		const struct FFIHIRExpr * e2;
	};

	struct FFIHIRForwardFunctionDecl {
		const char * name;
		const struct FFIHIRExpr * e2;
	};

	struct FFIHIRReturn {
		struct FFIHIRValue res;
	};

	union ExpressionUnion {
		struct FFIHIRVariableDecl variable_decl;
		struct FFIHIRFunctionDecl func_decl;
		struct FFIHIRForwardFunctionDecl forward_func_decl;
		uint8_t noop;
		struct FFIHIRReturn ret;
	};

	struct FFIHIRExpr {
		FFIHIRTag tag;
		ExpressionUnion value;
	};
}