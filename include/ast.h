#pragma once

#include <stdint.h>
#include <string>
#include <stddef.h>

extern "C" {
	struct FFIString {
		const char * data;
		size_t size;
	};

	struct FFIHIRArray {
		const struct FFIHIRValue * vals;
		size_t size;
	};

	struct FFIHIRInteger {
		size_t value;
	};

	struct FFIHIRFunctionCall {
		struct FFIString func_name;
		const struct FFIHIRValue * params;
		size_t param_len;
	};

	struct FFIHIRVariableReference {
		struct FFIString name;
	};

	union ValueUnion {
		struct FFIHIRArray array;
		struct FFIHIRInteger integer;
		struct FFIHIRVariableReference variable_reference;
		struct FFIHIRFunctionCall function_call;
	};

	enum FFIHirValueTag {
		Array = 0,
		Integer = 1,
		VariableReference = 2,
		FunctionCall = 3
	};

	struct FFIHIRValue {
		enum FFIHirValueTag tag;
		union ValueUnion value;
	};

	enum FFIHIRTag {
		VariableDecl = 0,
		Noop = 1,
		FunctionDecl = 2,
		ForwardFunctionDecl = 3,
		Return = 4,
	};

	struct FFIHIRVariableDecl {
		struct FFIString name;
		struct FFIHIRValue e1;
		struct FFIHIRExpr * e2;
	};

	struct FFIHIRFunctionDecl {
		struct FFIString name;
		const struct FFIHIRExpr * block;
		const struct FFIString * arg_names;
		size_t arg_len;
		const struct FFIHIRExpr * e2;
	};

	struct FFIHIRForwardFunctionDecl {
		struct FFIString name;
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