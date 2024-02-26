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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
	void codegen(FFIHIRExpr expression) {
		switch (expression.tag) {
		case VariableDecl:
			scad_constant(expression.value.variable_decl);
			break;
		case FunctionDecl:
			scad_func(expression.value.func_decl);
			break;
		case Noop:
			break;
		case Return:
			scad_return(expression.value.ret);
			break;
		case Yield:
			std::cout << "HERE IY" << std::endl;
			scad_yield(expression.value.yld);
			break;
		case ForwardFunctionDecl:
			scad_func_prototype(expression.value.forward_func_decl);
			break;
		case For:
			scad_for(expression.value.floop);
			break;
		default:
			std::cout
				<< " " << expression.tag
				<< "what are you trying to do to me lol expr \n\n\n";
			break;
		}
	}

	mlir::Value codegen(FFIHIRValue value) {
		switch (value.tag) {
		case Tensor:
			return scad_vector(value.value.tensor);
		case Integer:
			return scad_integer(value.value.integer);
		case VariableReference: {
			std::string variable_name(
				value.value.variable_reference.name.data,
				value.value.variable_reference.name.size
			);
			std::cout << variable_name << "ref" << std::endl;

			return variables[variable_name];
		}
		case FunctionCall:
			return scad_function_call(value.value.function_call);
		case Conditional:
			return scad_conditional(value.value.conditional);
			break;
		case Cast: 
			return scad_cast(value.value.cast);
		default:
			std::cout
				<< " " << value.tag
				<< "what are you trying to do to me lol val \n\n\n";
			break;
		}
	}

    private:
	mlir::MLIRContext & context;
	mlir::OpBuilder & builder;
	mlir::ModuleOp & mod;

	// Messages to passdown funtions
	bool is_generating_main = false;
	bool should_force_index_type = false;

	void * type_query_engine;

	std::unordered_map<std::string, mlir::Type> function_results;
	std::unordered_map<std::string, mlir::Value> variables;

	std::unordered_map<std::string, Alloc> allocations;

    private:
	FFIType query_type(std::string name) {
		return query(type_query_engine, name.data());
	}

	mlir::Type get_magnitude_type_for(FFIApplication t) {
		std::string tname(t.c.data, t.c.size);
		if (tname == "i32") {
			return builder.getI32Type();
		} else if (tname == "f32") {
			return builder.getF32Type();
		} else if (tname == "i64") {
			return builder.getI64Type();
		} else if (tname == "i16") {
			return builder.getI16Type();
		} else if (tname == "ii") {
			return builder.getIndexType();
		} else {
			std::cout << "SOMETHING WENT REALLY WRONG" << std::endl;
		}
	}

	mlir::Type get_type_for(FFIApplication t) {
		std::vector<int64_t> dims = get_dims_for(t);
		mlir::Type type = get_magnitude_type_for(t);

		if (dims.size() == 0) {
			return type;
		} else {
			return mlir::MemRefType::get(dims, type);
		}
	}

	std::vector<int64_t> get_dims_for(FFIApplication t) {
		if (t.dimensions_count == 0) {
			return std::vector<int64_t>();
		}

		std::vector<int64_t> dims(
			t.dimensions, t.dimensions + t.dimensions_count
		);

		return dims;
	}

	mlir::LogicalResult declare(std::string var, mlir::Value value) {
		if (variables.find(var) != variables.end()) {
			return mlir::failure();
		}
		variables[var] = value;
		return mlir::success();
	}

	mlir::MemRefType
	create_memref_type(mlir::ArrayRef<int64_t> shape, mlir::Type type) {
		return mlir::MemRefType::get(shape, type);
	}

	// mlir::DenseIntElementsAttr scad_matrix(FFIHIRTensor arr) {
	// 	std::vector<uint32_t> data;

	// 	for (size_t i = 0; i < arr.size; i++) {
	// 		data.push_back((uint32_t)arr.vals[i].value.integer.value
	// 		);
	// 	}

	// 	// The type of this attribute is tensor of 64-bit floating-point with the
	// 	// shape of the literal.
	// 	mlir::Type elementType = builder.getI32Type();
	// 	auto dataType = mlir::RankedTensorType::get(
	// 		{ (long long)arr.size }, elementType
	// 	);
	// 	return mlir::DenseIntElementsAttr::get(
	// 		dataType, llvm::ArrayRef(data)
	// 	);
	// }

	mlir::Type get_type_for_int_width(uint32_t width) {
		switch (width) {
		case 8:
			return builder.getI8Type();
		case 16:
			return builder.getI16Type();
		case 32:
			return builder.getI32Type();
		case 64:
			return builder.getI64Type();
		case 1000:
			return builder.getIndexType();
		}
	}

	mlir::Value scad_integer(FFIHIRInteger i) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, "Integer literal", 100, 100
		);
		if (i.width == 1000) {
			return builder.create<mlir::arith::ConstantIndexOp>(
				location, i.value
			);
		} else {
			auto attr = mlir::IntegerAttr::get(
				get_type_for_int_width(i.width),
				mlir::APInt(i.width, i.value)
			);

			return builder.create<mlir::arith::ConstantOp>(
				location, attr
			);
		}
	}

	mlir::Value scad_cast(FFIHIRCast i) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, "Integer literal", 100, 100
		);

		auto value_to_cast = codegen(*i.value);
		return builder.create<mlir::arith::IndexCastOp>(
			location, get_magnitude_type_for(i.app), value_to_cast
		);
		
	}

	mlir::Value scad_vector(FFIHIRTensor arr) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context,
			"lololololol you though i would be helpful?!!?",
			100,
			100
		);
		SmallVector<mlir::Value, 8> values;
		for (size_t i = 0; i < arr.size; i++) {
			auto value_at_index = codegen(arr.vals[i]);
			values.push_back(value_at_index);
		}

		auto alloc = builder.create<mlir::memref::AllocOp>(
			location,
			create_memref_type(arr.size, values[0].getType())
		);
		// auto * parentBlock = alloc->getBlock();

		for (size_t i = 0; i < arr.size; i++) {
			SmallVector<mlir::Value> indices;
			indices.push_back(
				builder.create<mlir::arith::ConstantIndexOp>(
					location, i
				)
			);
			builder.create<mlir::affine::AffineStoreOp>(
				location,
				values[i],
				alloc,
				llvm::ArrayRef(indices)
			);
		}

		// alloc->moveBefore(&parentBlock->front());

		return alloc;
	}

	mlir::LogicalResult scad_set(FFIHIRFunctionCall fc) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, "index_op", 100, 100
		);

		// Codegen the operands first.
		auto array = codegen(fc.params[0]);
		auto index = codegen(fc.params[1]);
		auto value = codegen(fc.params[2]);
		SmallVector<mlir::Value, 2> indices;

		indices.push_back(index);

		builder.create<mlir::affine::AffineStoreOp>(
			location, value, array, llvm::ArrayRef(indices)
		);

		return mlir::success();
	}

	mlir::LogicalResult scad_for(FFIHIRForLoop floop) {
		mlir::Location location =
			mlir::FileLineColLoc::get(&context, "for", 100, 100);

		auto start_point = floop.start.value.integer.value;
		auto end_point = floop.end.value.integer.value;

		auto loop = builder.create<mlir::affine::AffineForOp>(
			location, start_point, end_point, 1
		);
		builder.setInsertionPointToStart(&loop.getRegion().front());

		std::string ivname(floop.iv.data, floop.iv.size);
		variables[ivname] = builder.create<mlir::arith::IndexCastOp>(
			location, builder.getI32Type(), loop.getInductionVar()
		);
		codegen(*floop.block);
		builder.setInsertionPointAfter(loop);

		codegen(*floop.e2);

		return mlir::success();
	}

	mlir::LogicalResult scad_parallel(FFIHIRForLoop floop) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, "parallel", 100, 100
		);

		auto start_point = floop.start.value.integer.value;
		auto end_point = floop.end.value.integer.value;

		llvm::SmallVector<mlir::Type, 4> types;
		llvm::ArrayRef<mlir::Attribute> reductions;
		llvm::SmallVector<mlir::AffineExpr, 4> lbounds = {
			builder.getAffineDimExpr(0) + start_point
		};
		llvm::SmallVector<mlir::AffineExpr, 4> ubounds = {
			builder.getAffineDimExpr(0) + end_point
		};
		SmallVector<int32_t, 4> lboundGroup = { 1 };
		SmallVector<int32_t, 4> uboundGroup = { 1 };
		SmallVector<int64_t, 4> steps = { 1 };
		SmallVector<mlir::Value, 4> operands;

		operands.push_back(builder.create<mlir::arith::ConstantIndexOp>(
			location, 0
		));

		operands.push_back(builder.create<mlir::arith::ConstantIndexOp>(
			location, 0
		));

		auto loop = builder.create<mlir::affine::AffineParallelOp>(
			location,
			types,
			builder.getArrayAttr(reductions),
			mlir::AffineMapAttr::get(mlir::AffineMap::get(
				1, 0, lbounds, builder.getContext()
			)),
			builder.getI32TensorAttr(lboundGroup),
			mlir::AffineMapAttr::get(mlir::AffineMap::get(
				1, 0, ubounds, builder.getContext()
			)),
			builder.getI32TensorAttr(uboundGroup),
			builder.getI64ArrayAttr(steps),
			operands
		);

		auto & loop_body = loop.getRegion();
		auto block = builder.createBlock(&loop_body);

		loop.getBody()->addArgument(builder.getIndexType(), location);

		builder.setInsertionPointToStart(block);

		// std::string ivname(floop.iv.data, floop.iv.size);
		// variables[ivname] = builder.create<mlir::arith::IndexCastOp>(
		// 	location, builder.getI32Type(), loop.getIVs()[0]
		// );

		codegen(*floop.block);

		builder.create<mlir::affine::AffineYieldOp>(location);

		builder.setInsertionPointAfter(loop);

		codegen(*floop.e2);

		return mlir::success();
	}

	mlir::Value scad_constant(FFIHIRVariableDecl decl) {
		std::string name(decl.name.data, decl.name.size);
		mlir::Location location =
			mlir::FileLineColLoc::get(&context, name, 100, 100);
		// auto r = builder.create<mlir::scad::VectorOp>(
		// 	location, scad_matrix(decl.e1.value.array)
		// );
		auto r = codegen(decl.e1);

		if (decl.e1.tag == Tensor) {
			Alloc alloc_flag;
			alloc_flag.freed = false;
			alloc_flag.val = r;
			allocations[name] = alloc_flag;
		}

		std::cout << name << std::endl;
		variables[name] = r;

		codegen(*decl.e2);
		return r;
	}

	mlir::Value scad_function_call(FFIHIRFunctionCall fc) {
		std::string fname(fc.func_name.data, fc.func_name.size);

		mlir::Location location = mlir::FileLineColLoc::get(
			&context, fname + "Call", 100, 100
		);

		if (fname[0] == '@') {
			return inbuilt_op(fname, fc);
		}
		// Codegen the operands first.
		SmallVector<mlir::Value, 4> operands;
		for (size_t i = 0; i < fc.param_len; i++) {
			auto arg = codegen(fc.params[i]);
			if (!arg)
				return nullptr;
			operands.push_back(arg);
		}

		return builder.create<mlir::scad::GenericCallOp>(
			location,
			// mlir::RankedTensorType::get({ 2 }, builder.getI32Type()),
			function_results[fname],
			mlir::SymbolRefAttr::get(builder.getContext(), fname),
			operands
		);
	}

	mlir::Value scad_conditional(FFIHIRConditional cond) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, std::string("cond"), 100, 100
		);

		auto bl = builder.create<mlir::scad::BoolOp>(
			location,
			mlir::IntegerType::get(
				builder.getContext(),
				1,
				mlir::IntegerType::SignednessSemantics::Signless
			),
			true
		);
		auto scond =
			builder.create<mlir::scad::ConditionalOp>(location, bl);

		mlir::Block & if_arm = scond.getIfArm().front();
		mlir::Block & else_arm = scond.getElseArm().front();

		builder.setInsertionPointToStart(&if_arm);
		{ codegen(*cond.if_arm.block); }

		builder.setInsertionPointToStart(&else_arm);
		{ codegen(*cond.else_arm); }

		builder.setInsertionPointAfter(scond);

		return scond;
	}

	mlir::LogicalResult scad_print(FFIHIRFunctionCall fc) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, std::string("pritops"), 100, 100
		);

		auto arg = codegen(fc.params[0]);
		if (!arg)
			return mlir::failure();

		builder.create<mlir::scad::PrintOp>(location, arg);
		return mlir::success();
	}

	mlir::LogicalResult scad_drop(FFIHIRFunctionCall fc) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, std::string("dropop"), 100, 100
		);
		// Currently drop assumes a variable reference.
		// auto arg = codegen(fc.params[0]);

		std::string vrname(
			fc.params[0].value.variable_reference.name.data,
			fc.params[0].value.variable_reference.name.size
		);
		if (allocations.find(vrname) != allocations.end()) {
			Alloc & alloced = allocations[vrname];
			if (!alloced.freed) {
				// builder.create<mlir::scad::DropOp>(location, alloced.val);
				builder.create<mlir::memref::DeallocOp>(
					location, alloced.val
				);
				alloced.freed = true;
			}
		}

		return mlir::success();
	}

	void scad_func_prototype(FFIHIRForwardFunctionDecl ffd) {
		codegen(*ffd.e2);
	}

	mlir::Value inbuilt_op(std::string & name, FFIHIRFunctionCall fc) {
		if (name == "@print") {
			scad_print(fc);
			return nullptr;
		} else if (name == "@add") {
			return scad_add(fc);
		} else if (name.compare(0, 6, "@index") == 0) {
			// Its an index op
			return scad_index(fc);
		} else if (name == "@drop") {
			scad_drop(fc);
			return nullptr;
		} else if (name.compare(0, 4, "@set") == 0) {
			// Its a set op
			scad_set(fc);
			return nullptr;
		}
	}

	mlir::Value scad_add(FFIHIRFunctionCall fc) {
		mlir::Location location =
			mlir::FileLineColLoc::get(&context, "add_op", 100, 100);

		// Codegen the operands first.
		SmallVector<mlir::Value, 4> operands;
		for (size_t i = 0; i < fc.param_len; i++) {
			auto arg = codegen(fc.params[i]);
			if (!arg)
				return nullptr;
			operands.push_back(arg);
		}

		return builder.create<mlir::scad::AddOp>(
			location, operands[0].getType(), operands[0], operands[1]
		);
	}

	mlir::Value scad_index(FFIHIRFunctionCall fc) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, "index_op", 100, 100
		);

		// Codegen the operands first.
		auto array = codegen(fc.params[0]);
		auto index = codegen(fc.params[1]);

		SmallVector<mlir::Value, 4> indicies;
		indicies.push_back(index);

		return builder.create<mlir::affine::AffineLoadOp>(
			location, array, indicies
		);
	}

	mlir::scad::FuncOp
	proto_gen(FFIHIRFunctionDecl ffd, FFIType function_type) {
		std::string name = std::string(ffd.name.data, ffd.name.size);
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, name + "PROTO", 100, 100
		);

		llvm::SmallVector<mlir::Type, 4> arg_types;
		std::cout << name << " ARGLEN - " << ffd.arg_len << std::endl;
		for (size_t i = 0; i < ffd.arg_len; i++) {
			auto type = get_type_for(function_type.apps[i]);
			type.dump();
			arg_types.push_back(type);
		}

		auto type = builder.getFunctionType(arg_types, std::nullopt);

		if (name != "main") {
			function_results[name] = get_type_for(
				function_type.apps[function_type.size - 1]
			);
		}

		return builder.create<mlir::scad::FuncOp>(location, name, type);
	}

	mlir::scad::FuncOp scad_func(FFIHIRFunctionDecl decl) {
		std::string name = std::string(decl.name.data, decl.name.size);

		mlir::Location location = mlir::FileLineColLoc::get(
			&context, name + " Decl", 100, 100
		);
		// Create an MLIR function for the given prototype.
		FFIType type = query_type(name);

		builder.setInsertionPointToEnd(mod.getBody());
		mlir::scad::FuncOp function = proto_gen(decl, type);

		mlir::Block & entryBlock = function.front();

		for (size_t i = 0; i < decl.arg_len; i++) {
			if (failed(
				    declare(std::string(
						    decl.arg_names[i].data,
						    decl.arg_names[i].size
					    ),
					    entryBlock.getArguments()[i])
			    )) {
				std::cout << "i failed you. srry";
				return nullptr;
			}
		}
		builder.setInsertionPointToStart(&entryBlock);
		// mlir::AutomaticAllocationScope allocationScope(builder);

		// Emit the body of the function.
		if (name == "main")
			is_generating_main = true;
		codegen(*decl.block);
		is_generating_main = false;
		builder.setInsertionPointToEnd(mod.getBody());

		mlir::scad::ReturnOp returnOp;
		if (!entryBlock.empty())
			returnOp = llvm::dyn_cast<mlir::scad::ReturnOp>(
				entryBlock.back()
			);
		if (!returnOp) {
			builder.create<mlir::scad::ReturnOp>(location);
		} else if (returnOp.hasOperand() && name != "main") {
			// Otherwise, if this return operation has an operand then add a result to
			// the function.
			auto rettype = get_type_for(type.apps[type.size - 1]);

			rettype.dump();
			function.setType(builder.getFunctionType(
				function.getFunctionType().getInputs(), rettype
			));
		}

		codegen(*decl.e2);
		// If this function isn't main, then set the visibility to private.
		// if (funcAST.getProto()->getName() != "main")
		// 	function.setPrivate();

		return function;
	}

	mlir::LogicalResult scad_return(FFIHIRReturn ret) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, std::string("RETURN STATEMENT!!!"), 100, 100
		);

		// std::string refer = std::string(
		// 	ret.res.value.variable_reference.name.data,
		// 	ret.res.value.variable_reference.name.size
		// );

		if (is_generating_main) {
			builder.create<mlir::scad::ReturnOp>(location);
		} else {
			auto ret_val = codegen(ret.res);
			builder.create<mlir::scad::ReturnOp>(
				location, ArrayRef(ret_val)
			);
		}

		return mlir::success();
	}

	mlir::LogicalResult scad_yield(FFIHIRYield yld) {
		mlir::Location location = mlir::FileLineColLoc::get(
			&context, std::string("YIELD STATEMENT!!!"), 100, 100
		);

		std::string refer = std::string(
			yld.res.value.variable_reference.name.data,
			yld.res.value.variable_reference.name.size
		);

		builder.create<mlir::scad::YieldOp>(
			location, ArrayRef(variables[refer])
		);
		return mlir::success();
	}
};

#endif