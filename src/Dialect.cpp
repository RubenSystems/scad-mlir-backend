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

/*
==
  Function Op 
==
*/

void FuncOp::build(
	mlir::OpBuilder & builder,
	mlir::OperationState & state,
	llvm::StringRef name,
	mlir::FunctionType type,
	llvm::ArrayRef<mlir::NamedAttribute> attrs
) {
	buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult
FuncOp::parse(mlir::OpAsmParser & parser, mlir::OperationState & result) {
	auto buildFuncType = [](mlir::Builder & builder,
				llvm::ArrayRef<mlir::Type> argTypes,
				llvm::ArrayRef<mlir::Type> results,
				mlir::function_interface_impl::VariadicFlag,
				std::string &) {
		return builder.getFunctionType(argTypes, results);
	};

	return mlir::function_interface_impl::parseFunctionOp(
		parser,
		result,
		/*allowVariadic=*/false,
		getFunctionTypeAttrName(result.name),
		buildFuncType,
		getArgAttrsAttrName(result.name),
		getResAttrsAttrName(result.name)
	);
}

void FuncOp::print(mlir::OpAsmPrinter & p) {
	mlir::function_interface_impl::printFunctionOp(
		p,
		*this,
		/*isVariadic=*/false,
		getFunctionTypeAttrName(),
		getArgAttrsAttrName(),
		getResAttrsAttrName()
	);
}

/*
== 
ReturnOp
==
*/

mlir::LogicalResult ReturnOp::verify() {
	return mlir::success();
}

/*
==
Call Op
==
*/

void GenericCallOp::build(
	mlir::OpBuilder & builder,
	mlir::OperationState & state,
	StringRef callee,
	ArrayRef<mlir::Value> arguments
) {
	state.addTypes(UnrankedTensorType::get(builder.getI32Type()));
	state.addOperands(arguments);
	state.addAttribute(
		"callee", mlir::SymbolRefAttr::get(builder.getContext(), callee)
	);
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
	return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
	(*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() {
	return getInputs();
}

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
	return getInputsMutable();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"