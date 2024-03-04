#ifndef SCADC_MLIR_GENERATOR_H
#define SCADC_MLIR_GENERATOR_H
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "Dialect.h"
#include "ast.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

struct Alloc {
	mlir::Value val;
	bool freed;
};

class SCADMIRLowering {
    public:
	SCADMIRLowering(
		mlir::MLIRContext & context,
		mlir::OpBuilder & builder,
		mlir::ModuleOp & mod,
		void * type_query_engine
	)
		: context(context)
		, builder(builder)
		, mod(mod)
		, type_query_engine(type_query_engine) {
	}

    public:
	void codegen(FFIHIRExpr expression);

	mlir::Value codegen(FFIHIRValue value);

    private:
	mlir::MLIRContext & context;
	mlir::OpBuilder & builder;
	mlir::ModuleOp & mod;

	// Messages to passdown funtions
	bool is_generating_main = false;

	void * type_query_engine;

	std::unordered_map<std::string, mlir::Type> function_results;
	std::unordered_map<std::string, mlir::Value> variables;

	std::unordered_map<std::string, Alloc> allocations;

    private:
	FFIType query_type(std::string name) {
		return query(type_query_engine, name.data());
	}

	mlir::Type get_magnitude_type_for(FFIApplication t);

	mlir::Type get_type_for(FFIApplication t);

	std::vector<int64_t> get_dims_for(FFIApplication t);

	mlir::LogicalResult declare(std::string var, mlir::Value value);

	mlir::MemRefType
	create_memref_type(mlir::ArrayRef<int64_t> shape, mlir::Type type);

	mlir::Type get_type_for_int_width(uint32_t width);

	mlir::Value scad_integer(FFIHIRInteger i);

	mlir::Value scad_cast(FFIHIRCast i);

	mlir::Value scad_vector(FFIHIRTensor arr);

	mlir::LogicalResult scad_set(FFIHIRFunctionCall fc);

	mlir::LogicalResult scad_for(FFIHIRForLoop floop);

	mlir::LogicalResult scad_parallel(FFIHIRForLoop floop);

	mlir::Value scad_constant(FFIHIRVariableDecl decl);

	mlir::Value scad_function_call(FFIHIRFunctionCall fc);

	mlir::Value scad_conditional(FFIHIRConditional cond);

	mlir::LogicalResult scad_print(FFIHIRFunctionCall fc);

	mlir::LogicalResult scad_drop(FFIHIRFunctionCall fc);

	void scad_func_prototype(FFIHIRForwardFunctionDecl ffd);

	mlir::Value inbuilt_op(std::string & name, FFIHIRFunctionCall fc);

	template <typename Operation>
	mlir::Value scad_scalar_op(FFIHIRFunctionCall fc);

	mlir::Value scad_vector_load_op(FFIHIRFunctionCall fc);
	mlir::LogicalResult scad_vector_store_op(FFIHIRFunctionCall fc);

	template <typename Operation>
	mlir::Value scad_vectorised_op(FFIHIRFunctionCall fc);

	mlir::Value scad_index(FFIHIRFunctionCall fc);

	mlir::scad::FuncOp
	proto_gen(FFIHIRFunctionDecl ffd, FFIType function_type);

	mlir::scad::FuncOp scad_func(FFIHIRFunctionDecl decl);

	mlir::LogicalResult scad_return(FFIHIRReturn ret);

	mlir::LogicalResult scad_yield(FFIHIRYield yld);
};

#endif