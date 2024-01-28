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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

mlir::DenseIntElementsAttr generate_mlir_mat(mlir::OpBuilder & builder) {
	std::vector<uint32_t> data = { 100, 200, 300, 400, 500, 600 };

	// The type of this attribute is tensor of 64-bit floating-point with the
	// shape of the literal.
	mlir::Type elementType = builder.getI32Type();
	auto dataType = mlir::RankedTensorType::get({ 3, 2 }, elementType);
	return mlir::DenseIntElementsAttr::get(dataType, llvm::ArrayRef(data));
}

mlir::Value generate_mlir_constant(
	std::string name,
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, name, 100, 100);

	// This is the actual attribute that holds the list of values for this
	// tensor literal.
	return builder.create<mlir::scad::VectorOp>(
		location, generate_mlir_mat(builder)
	);
}

  mlir::LogicalResult generate_mlir_print(	
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context,
	mlir::Value arg

) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("pritops"), 100, 100
	);

    builder.create<mlir::scad::PrintOp>(location, arg);
    return mlir::success();
  }

mlir::scad::FuncOp generate_mlir_func_proto(
	std::string name,
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("main c"), 100, 100
	);

	// This is a generic function, the return type will be inferred later.
	// Arguments type are uniformly unranked tensors.
	llvm::SmallVector<mlir::Type, 4> argTypes(
		0, mlir::RankedTensorType::get(10, builder.getI32Type())
	);
	auto funcType = builder.getFunctionType(argTypes, std::nullopt);

	builder.setInsertionPointToEnd(mod.getBody());
	return builder.create<mlir::scad::FuncOp>(location, name, funcType);
}

mlir::Value generate_mlir_function_call(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	llvm::StringRef callee = "do_something";
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("main d"), 100, 100
	);

	// Codegen the operands first.
	SmallVector<mlir::Value, 4> operands;
	// for (auto & expr : call.getArgs()) {
	// 	auto arg = mlirGen(*expr);
	// 	if (!arg)
	// 		return nullptr;
	// operands.push_back(
	// 	generate_mlir_constant("v1 array", builder, mod, context)
	// );
	// }

	return builder.create<mlir::scad::GenericCallOp>(
		location,
		mlir::RankedTensorType::get({ 3, 2 }, builder.getI32Type()),
		mlir::SymbolRefAttr::get(builder.getContext(), callee),
		operands
	);
}

mlir::LogicalResult generate_mlir_return(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context,
	mlir::Value expr
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("do_something a"), 100, 100
	);

	builder.create<mlir::scad::ReturnOp>(location, ArrayRef(expr));
	return mlir::success();
}

mlir::scad::FuncOp generate_mlir_func_v2(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("do_something b"), 100, 100
	);
	// Create an MLIR function for the given prototype.
	builder.setInsertionPointToEnd(mod.getBody());
	mlir::scad::FuncOp function = generate_mlir_func_proto(
		std::string("do_something"), builder, mod, context
	);
	if (!function)
		return nullptr;

	// Let's start the body of the function now!
	mlir::Block & entryBlock = function.front();
	// llvm::ArrayRef<std::unique_ptr<ExprAST>> protoArgs = funcAST.getProto()->getArgs();

	// Declare all the function arguments in the symbol table.
	// for (const auto nameValue :
	//      llvm::zip(protoArgs, entryBlock.getArguments())) {
	//   if (failed(declare(std::get<0>(nameValue)->getName(),
	//                      std::get<1>(nameValue))))
	//     return nullptr;
	// }

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Emit the body of the function.
	generate_mlir_constant("V2 array 1", builder, mod, context);
	auto expression_to_ret =
		generate_mlir_constant("V2 array2", builder, mod, context);
	generate_mlir_return(builder, mod, context, expression_to_ret);
	// Implicitly return void if no return statement was emitted.
	// FIXME: we may fix the parser instead to always return the last expression
	// (this would possibly hel∂=p the REPL case later)
	mlir::scad::ReturnOp returnOp;
	if (!entryBlock.empty())
		returnOp =
			llvm::dyn_cast<mlir::scad::ReturnOp>(entryBlock.back());
	if (!returnOp) {
		builder.create<mlir::scad::ReturnOp>(location);
	} else if (returnOp.hasOperand()) {
		// Otherwise, if this return operation has an operand then add a result to
		// the function.
		function.setType(builder.getFunctionType(
			function.getFunctionType().getInputs(),
			mlir::RankedTensorType::get(
				{ 3, 2 }, builder.getI32Type()
			)
		));
	}

	// If this function isn't main, then set the visibility to private.
	// if (funcAST.getProto()->getName() != "main")
	// 	function.setPrivate();

	return function;
}

mlir::scad::FuncOp generate_mlir_func(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("main a"), 100, 100
	);
	// Create an MLIR function for the given prototype.
	builder.setInsertionPointToEnd(mod.getBody());
	mlir::scad::FuncOp function = generate_mlir_func_proto(
		std::string("main"), builder, mod, context
	);
	if (!function)
		return nullptr;

	// Let's start the body of the function now!
	mlir::Block & entryBlock = function.front();
	// llvm::ArrayRef<std::unique_ptr<ExprAST>> protoArgs = funcAST.getProto()->getArgs();

	// Declare all the function arguments in the symbol table.
	// for (const auto nameValue :
	//      llvm::zip(protoArgs, entryBlock.getArguments())) {
	//   if (failed(declare(std::get<0>(nameValue)->getName(),
	//                      std::get<1>(nameValue))))
	//     return nullptr;
	// }

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Emit the body of the function.
	generate_mlir_constant("v1 array", builder, mod, context);
	generate_mlir_print(builder, mod, context, generate_mlir_constant("v1 array1", builder, mod, context));
	generate_mlir_function_call(builder, mod, context);
	// Implicitly return void if no return statement was emitted.
	// FIXME: we may fix the parser instead to always return the last expression
	// (this would possibly help the REPL case later)
	mlir::scad::ReturnOp returnOp;
	if (!entryBlock.empty())
		returnOp =
			llvm::dyn_cast<mlir::scad::ReturnOp>(entryBlock.back());
	if (!returnOp) {
		builder.create<mlir::scad::ReturnOp>(location);
	} else if (returnOp.hasOperand()) {
		// Otherwise, if this return operation has an operand then add a result to
		// the function.
		function.setType(builder.getFunctionType(
			function.getFunctionType().getInputs(),
			mlir::RankedTensorType::get(10, builder.getI32Type())
		));
	}

	// If this function isn't main, then set the visibility to private.
	// if (funcAST.getProto()->getName() != "main")
	// 	function.setPrivate();

	return function;
}

#endif