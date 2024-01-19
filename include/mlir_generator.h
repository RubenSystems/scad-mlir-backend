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

mlir::Value generate_mlir_constant(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("main"), 100, 100
	);
	return builder.create<mlir::scad::VectorOp>(location, 10);
}

mlir::scad::FuncOp generate_mlir_func_proto(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("main"), 100, 100
	);

	// This is a generic function, the return type will be inferred later.
	// Arguments type are uniformly unranked tensors.
	llvm::SmallVector<mlir::Type, 4> argTypes(
		0, mlir::RankedTensorType::get(10, builder.getI32Type())
	);
	auto funcType = builder.getFunctionType(argTypes, std::nullopt);

	builder.setInsertionPointToEnd(mod.getBody());
	return builder.create<mlir::scad::FuncOp>(location, "main", funcType);
}

mlir::scad::FuncOp generate_mlir_func(
	mlir::OpBuilder & builder,
	mlir::ModuleOp & mod,
	mlir::MLIRContext & context
) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("main"), 100, 100
	);
	// Create an MLIR function for the given prototype.
	builder.setInsertionPointToEnd(mod.getBody());
	mlir::scad::FuncOp function =
		generate_mlir_func_proto(builder, mod, context);
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
	generate_mlir_constant(builder, mod, context);

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