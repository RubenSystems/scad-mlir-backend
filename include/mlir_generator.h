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

mlir::Value generate_mlir_constant(mlir::MLIRContext &context, mlir::ModuleOp & mod, mlir::OpBuilder & builder) {

	mlir::Location location = mlir::FileLineColLoc::get(&context, std::string("jeef"), 100, 100);
	return builder.create<mlir::scad::VectorOp>(location, 10);
}

mlir::OwningOpRef<mlir::ModuleOp> generate_mlir(mlir::MLIRContext &context) {
	mlir::OpBuilder builder(&context);
	mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());
	builder.setInsertionPointToEnd(mod.getBody());
	generate_mlir_constant(context, mod, builder);

	return mod;
}

#endif