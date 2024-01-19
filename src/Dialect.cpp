#include "Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::scad;

#include "Dialect.cpp.inc"

void SCADDialect::initialize() {
	addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
		>();
}

/*
===
  Vector Op
===
*/
void VectorOp::build(
	mlir::OpBuilder & builder,
	mlir::OperationState & state,
	int32_t value
) {
	auto dataType = RankedTensorType::get({}, builder.getI32Type());
	auto dataAttribute = DenseIntElementsAttr::get(dataType, value);
	VectorOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult
VectorOp::parse(mlir::OpAsmParser & parser, mlir::OperationState & result) {
	mlir::DenseElementsAttr value;
	if (parser.parseOptionalAttrDict(result.attributes) ||
	    parser.parseAttribute(value, "value", result.attributes))
		return failure();

	result.addTypes(value.getType());
	return success();
}

void VectorOp::print(mlir::OpAsmPrinter & printer) {
	printer << " ";
	printer.printOptionalAttrDict(
		(*this)->getAttrs(),
		/*elidedAttrs=*/{ "value" }
	);
	printer << getValue();
}

mlir::LogicalResult VectorOp::verify() {
	return mlir::success();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"