#pragma once

#include <stdint.h>
#include <string>
#include <stddef.h>

// AST Types
extern "C" {
	struct FFIString {
		const char * data;
		size_t size;
	};

	struct FFIApplication {
		struct FFIString c;
		const uint32_t * dimensions;
		size_t dimensions_count;
	};

	struct FFIType {
		size_t size;
		struct FFIApplication * apps;
	};

	struct FFIHIRTensor {
		const struct FFIHIRValue * vals;
		size_t size;
	};

	struct FFIHIRInteger {
		size_t value;
		uint32_t width;
	};

	struct FFIHIRFunctionCall {
		struct FFIString func_name;
		const struct FFIHIRValue * params;
		size_t param_len;
	};

	struct FFIHIRVariableReference {
		struct FFIString name;
	};

	struct FFIExpressionBlock {
		struct FFIHIRValue * condition;
		struct FFIHIRExpr * block;
	};

	struct FFIHIRConditional {
		struct FFIExpressionBlock if_arm;
		struct FFIHIRExpr * else_arm;
	};

	struct FFIHIRCast {
		struct FFIHIRValue * value; 
		struct FFIApplication app; 
	};

	union ValueUnion {
		struct FFIHIRTensor tensor;
		struct FFIHIRInteger integer;
		struct FFIHIRVariableReference variable_reference;
		struct FFIHIRFunctionCall function_call;
		uint8_t boolean;
		struct FFIHIRConditional conditional;
		struct FFIHIRCast cast;
	};

	enum FFIHirValueTag {
		Tensor = 0,
		Integer = 1,
		VariableReference = 2,
		FunctionCall = 3,
		Bool = 4,
		Conditional = 5,
		Float = 6,
		Cast = 7
	};

	struct FFIHIRValue {
		enum FFIHirValueTag tag;
		union ValueUnion value;
	};

	struct FFIHIRForLoop {
		struct FFIString iv;
		struct FFIHIRValue start;
		struct FFIHIRValue end;
		const struct FFIHIRExpr * block;
		bool parallel;
		const struct FFIHIRExpr * e2;
	};

	enum FFIHIRTag {
		VariableDecl = 0,
		Noop = 1,
		FunctionDecl = 2,
		ForwardFunctionDecl = 3,
		Return = 4,
		Yield = 5,
		For = 6
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

	struct FFIHIRYield {
		struct FFIHIRValue res;
	};

	union ExpressionUnion {
		struct FFIHIRVariableDecl variable_decl;
		struct FFIHIRFunctionDecl func_decl;
		struct FFIHIRForwardFunctionDecl forward_func_decl;
		uint8_t noop;
		struct FFIHIRReturn ret;
		struct FFIHIRYield yld;
		struct FFIHIRForLoop floop;
	};

	struct FFIHIRExpr {
		FFIHIRTag tag;
		ExpressionUnion value;
	};
}

// functionality
extern "C" {
	struct FFIHIRExpr compile(const char *, void **);
	struct FFIType query(void *, const char *);
}