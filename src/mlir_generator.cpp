

#include <mlir_generator.h>
#include <iostream>

void SCADMIRLowering::codegen(FFIHIRExpr expression) {
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
		scad_yield(expression.value.yld);
		break;
	case ForwardFunctionDecl:
		scad_func_prototype(expression.value.forward_func_decl);
		break;
	case For:
		if (expression.value.floop.parallel) {
			scad_parallel(expression.value.floop);
		} else {
			scad_for(expression.value.floop);
		}
		break;
	case While:
		scad_while(expression.value.whl);
		break;
	default:
		std::cout << " " << expression.tag
			  << "what are you trying to do to me lol expr \n\n\n";
		break;
	}
}

mlir::Value SCADMIRLowering::codegen(FFIHIRValue value) {
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
		std::cout << variable_name << " ref" << std::endl;
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
		std::cout << " " << value.tag
			  << "what are you trying to do to me lol val \n\n\n";
		break;
	}
}

mlir::Type SCADMIRLowering::get_magnitude_type_for(FFIApplication t) {
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

mlir::Type SCADMIRLowering::get_type_for(FFIApplication t) {
	std::vector<int64_t> dims = get_dims_for(t);
	mlir::Type type = get_magnitude_type_for(t);

	if (dims.size() == 0) {
		return type;
	} else {
		return mlir::MemRefType::get(dims, type);
	}
}

std::vector<int64_t> SCADMIRLowering::get_dims_for(FFIApplication t) {
	if (t.dimensions_count == 0) {
		return std::vector<int64_t>();
	}

	std::vector<int64_t> dims(
		t.dimensions, t.dimensions + t.dimensions_count
	);

	return dims;
}

mlir::LogicalResult
SCADMIRLowering::declare(std::string var, mlir::Value value) {
	if (variables.find(var) != variables.end()) {
		return mlir::failure();
	}
	variables[var] = value;
	return mlir::success();
}

mlir::MemRefType SCADMIRLowering::create_memref_type(
	mlir::ArrayRef<int64_t> shape,
	mlir::Type type
) {
	return mlir::MemRefType::get(shape, type);
}

mlir::Type SCADMIRLowering::get_type_for_int_width(uint32_t width) {
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
	default:
		std::cout << "FAILURE GET TYPE FOR WIDTH" << std::endl;
	}
}

mlir::Value SCADMIRLowering::scad_integer(FFIHIRInteger i) {
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

		return builder.create<mlir::arith::ConstantOp>(location, attr);
	}
}

mlir::Value SCADMIRLowering::scad_cast(FFIHIRCast i) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, "Integer literal", 100, 100
	);

	auto value_to_cast = codegen(*i.value);
	return builder.create<mlir::arith::IndexCastOp>(
		location, get_magnitude_type_for(i.app), value_to_cast
	);
}

mlir::Value SCADMIRLowering::scad_vector(FFIHIRTensor arr) {
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
		location, create_memref_type(arr.size, values[0].getType())
	);
	// auto * parentBlock = alloc->getBlock();

	for (size_t i = 0; i < arr.size; i++) {
		SmallVector<mlir::Value> indices;
		indices.push_back(builder.create<mlir::arith::ConstantIndexOp>(
			location, i
		));
		builder.create<mlir::memref::StoreOp>(
			location, values[i], alloc, llvm::ArrayRef(indices)
		);
	}

	// alloc->moveBefore(&parentBlock->front());

	return alloc;
}

mlir::LogicalResult SCADMIRLowering::scad_set(FFIHIRFunctionCall fc) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "index_op", 100, 100);

	// Codegen the operands first.
	auto array = codegen(fc.params[0]);
	auto index = codegen(fc.params[1]);
	auto value = codegen(fc.params[2]);
	SmallVector<mlir::Value, 2> indices;

	indices.push_back(index);

	builder.create<mlir::memref::StoreOp>(
		location, value, array, llvm::ArrayRef(indices)
	);

	return mlir::success();
}

mlir::LogicalResult SCADMIRLowering::scad_for(FFIHIRForLoop floop) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "for", 100, 100);

	auto start_point = codegen(floop.start);
	auto end_point = codegen(floop.end);
	auto step = codegen(floop.step);

	auto loop = builder.create<mlir::scf::ForOp>(
		location, start_point, end_point, step
	);
	std::string ivname(floop.iv.data, floop.iv.size);
	variables[ivname] = loop.getInductionVar();
	std::cout << ivname << " iv" << std::endl;

	builder.setInsertionPointToStart(&loop.getRegion().front());
	codegen(*floop.block);
	builder.setInsertionPointAfter(loop);

	codegen(*floop.e2);

	return mlir::success();
}

mlir::LogicalResult SCADMIRLowering::scad_while(FFIHIRWhile whl) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "while", 100, 100);

	llvm::SmallVector<mlir::Value> ops = {};
	llvm::SmallVector<mlir::Type> types;
	auto loop = builder.create<mlir::scf::WhileOp>(
		location,
		types,
		ops,
		[&](mlir::OpBuilder & builder,
		    mlir::Location loc,
		    mlir::ValueRange cond) {
			// Generate condition depends code
			codegen(*whl.cond_expr);
			// Generate condition
			auto condition = codegen(whl.condition);
			llvm::SmallVector<mlir::Value> ops = {};
			builder.create<mlir::scf::ConditionOp>(
				loc, condition, ops
			);
		},
		[&](mlir::OpBuilder & builder,
		    mlir::Location loc,
		    mlir::ValueRange cond) {
			codegen(*whl.block);

			llvm::SmallVector<mlir::Value> ops = {};

			builder.create<mlir::scf::YieldOp>(loc, ops);
		}
	);

	codegen(*whl.e2);

	return mlir::success();
}

mlir::LogicalResult SCADMIRLowering::scad_parallel(FFIHIRForLoop floop) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "parallel", 100, 100);

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

	operands.push_back(
		builder.create<mlir::arith::ConstantIndexOp>(location, 0)
	);

	operands.push_back(
		builder.create<mlir::arith::ConstantIndexOp>(location, 0)
	);

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

	std::string ivname(floop.iv.data, floop.iv.size);
	variables[ivname] = loop.getIVs()[0];

	builder.setInsertionPointToStart(block);

	codegen(*floop.block);
	builder.create<mlir::affine::AffineYieldOp>(location);

	builder.setInsertionPointAfter(loop);

	codegen(*floop.e2);

	return mlir::success();
}

mlir::Value SCADMIRLowering::scad_constant(FFIHIRVariableDecl decl) {
	std::string name(decl.name.data, decl.name.size);

	// auto r = builder.create<mlir::scad::VectorOp>(
	// 	location, scad_matrix(decl.e1.value.array)
	// );
	auto r = codegen(decl.e1);

	// if (decl.e1.tag == Tensor) {
	// 	Alloc alloc_flag;
	// 	alloc_flag.freed = false;
	// 	alloc_flag.val = r;
	// 	allocations[name] = alloc_flag;
	// }
	variables[name] = r;

	codegen(*decl.e2);
	return r;
}

mlir::Value SCADMIRLowering::scad_function_call(FFIHIRFunctionCall fc) {
	std::string fname(fc.func_name.data, fc.func_name.size);

	mlir::Location location =
		mlir::FileLineColLoc::get(&context, fname + "Call", 100, 100);

	if (fname[0] == '@') {
		return inbuilt_op(fname, fc);
	}
	// Codegen the operands first.
	SmallVector<mlir::Value, 4> operands;
	for (size_t i = 0; i < fc.param_len; i++) {
		auto parse_arg = fc.params[i];
		auto arg = codegen(parse_arg);
		if (!arg) {
			std::cout << "unable to codegen arg " << i << " for "
				  << fname << std::endl;
			return nullptr;
		}
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

mlir::Value SCADMIRLowering::scad_conditional(FFIHIRConditional cond) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("cond"), 100, 100
	);

	auto condition = codegen(*cond.if_arm.condition);

	auto scond = builder.create<mlir::scad::ConditionalOp>(
		location, condition, builder.getI32Type()
	);

	mlir::Block & if_arm = scond.getIfArm().front();
	mlir::Block & else_arm = scond.getElseArm().front();

	builder.setInsertionPointToStart(&if_arm);
	{ codegen(*cond.if_arm.block); }

	builder.setInsertionPointToStart(&else_arm);
	{ codegen(*cond.else_arm); }

	builder.setInsertionPointAfter(scond);

	return scond;
}

mlir::LogicalResult SCADMIRLowering::scad_print(FFIHIRFunctionCall fc) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("pritops"), 100, 100
	);

	auto arg = codegen(fc.params[0]);
	if (!arg)
		return mlir::failure();

	builder.create<mlir::scad::PrintOp>(location, arg);
	return mlir::success();
}

mlir::LogicalResult SCADMIRLowering::scad_drop(FFIHIRFunctionCall fc) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("dropop"), 100, 100
	);
	// Currently drop assumes a variable reference.
	auto arg = codegen(fc.params[0]);

	auto type = arg.getType();
	if (mlir::MemRefType memRefType = type.dyn_cast_or_null<mlir::MemRefType>()) {
		builder.create<mlir::memref::DeallocOp>(location, arg);
	}

	return mlir::success();
}

void SCADMIRLowering::scad_func_prototype(FFIHIRForwardFunctionDecl ffd) {
	codegen(*ffd.e2);
}

mlir::Value
SCADMIRLowering::inbuilt_op(std::string & name, FFIHIRFunctionCall fc) {
	if (name == "@print") {
		scad_print(fc);
		return nullptr;
	} else if (name == "@add") {
		return scad_scalar_op<mlir::arith::AddIOp>(fc);
	} else if (name == "@div") {
		return scad_scalar_op<mlir::arith::DivSIOp>(fc);
	} else if (name == "@lt") {
		return scad_cmp_op(fc, mlir::arith::CmpIPredicate::slt);
	} else if (name == "@lte") {
		return scad_cmp_op(fc, mlir::arith::CmpIPredicate::sle);
	} else if (name == "@eq") {
		return scad_cmp_op(fc, mlir::arith::CmpIPredicate::eq);
	} else if (name == "@gte") {
		return scad_cmp_op(fc, mlir::arith::CmpIPredicate::sge);
	} else if (name == "@gt") {
		return scad_cmp_op(fc, mlir::arith::CmpIPredicate::sgt);
	} else if (name == "@sub") {
		return scad_scalar_op<mlir::arith::SubIOp>(fc);
	} else if (name == "@mul") {
		return scad_scalar_op<mlir::arith::MulIOp>(fc);
	} else if (name == "@add.v") {
		return scad_vectorised_op<mlir::arith::AddIOp>(fc);
	} else if (name == "@sub.v") {
		return scad_vectorised_op<mlir::arith::SubIOp>(fc);
	} else if (name == "@mul.v") {
		return scad_vectorised_op<mlir::arith::MulIOp>(fc);
	} else if (name == "@prefetch.read") {
		scad_prefetch_read(fc);
		return nullptr;
	} else if (name == "@prefetch.write") {
		scad_prefetch_write(fc);
		return nullptr;
	} else if (name == "@vec.store") {
		scad_vector_store_op(fc);
		return nullptr;
	} else if (name == "@empty") {
		return scad_empty(fc);
	} else if (name == "@vec.load") {
		return scad_vector_load_op(fc);
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

mlir::Value SCADMIRLowering::scad_empty(FFIHIRFunctionCall fc) {
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
	auto size = llvm::cast<mlir::arith::ConstantIndexOp>(
		operands[1].getDefiningOp()
	);

	auto splat = builder.create<mlir::vector::SplatOp>(
		location,
		mlir::VectorType::get({ size.value() }, operands[0].getType()),
		operands[0]
	);

	auto alloc = builder.create<mlir::memref::AllocOp>(
		location,
		create_memref_type({ size.value() }, operands[0].getType())
	);

	llvm::SmallVector<mlir::Value> offset = {
		builder.create<mlir::arith::ConstantIndexOp>(
			mlir::UnknownLoc::get(&context), 0
		)
	};
	builder.create<mlir::vector::StoreOp>(
		mlir::UnknownLoc::get(&context),
		splat /*vec to laod from*/,
		alloc /*memref to store to*/,
		offset /*Offset*/
	);

	return alloc;
}

template <typename Operation>
mlir::Value SCADMIRLowering::scad_scalar_op(FFIHIRFunctionCall fc) {
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

	return builder.create<Operation>(
		location, operands[0].getType(), operands[0], operands[1]
	);
}

mlir::Value SCADMIRLowering::scad_cmp_op(
	FFIHIRFunctionCall fc,
	mlir::arith::CmpIPredicate comparitor
) {
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

	return builder.create<mlir::arith::CmpIOp>(
		location,
		builder.getI1Type(),
		comparitor,
		operands[0],
		operands[1]
	);
}

mlir::Value SCADMIRLowering::scad_vector_load_op(FFIHIRFunctionCall fc) {
	SmallVector<mlir::Value, 4> operands;
	for (size_t i = 0; i < fc.param_len; i++) {
		auto arg = codegen(fc.params[i]);
		if (!arg) {
			std::cout
				<< "failed to parse arg in scad vectorised add";
		}
		operands.push_back(arg);
	}

	auto size = llvm::cast<mlir::arith::ConstantIndexOp>(
		operands[2].getDefiningOp()
	);
	llvm::SmallVector<mlir::Value> load_ops = { operands[1] };
	llvm::SmallVector<mlir::Type> res_type = { mlir::VectorType::get(
		{ size.value() },
		llvm::cast<mlir::MemRefType>(operands[0].getType())
			.getElementType()
	)

	};

	return builder.create<mlir::vector::LoadOp>(
		mlir::UnknownLoc::get(&context), res_type, operands[0], load_ops
	);
}

mlir::LogicalResult SCADMIRLowering::scad_vector_store_op(FFIHIRFunctionCall fc
) {
	SmallVector<mlir::Value, 4> operands;
	for (size_t i = 0; i < fc.param_len; i++) {
		auto arg = codegen(fc.params[i]);
		if (!arg) {
			std::cout
				<< "failed to parse arg in scad vectorised add";
		}
		operands.push_back(arg);
	}

	builder.create<mlir::vector::StoreOp>(
		mlir::UnknownLoc::get(&context),
		operands[1] /*memref to store to*/,
		operands[0] /*vec to laod from*/,
		operands[2] /*Offset*/
	);
	return mlir::success();
}

template <typename Operation>
mlir::Value SCADMIRLowering::scad_vectorised_op(FFIHIRFunctionCall fc) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, "vectorised add op", 100, 100
	);

	SmallVector<mlir::Value, 4> operands;
	for (size_t i = 0; i < fc.param_len; i++) {
		auto arg = codegen(fc.params[i]);
		if (!arg) {
			std::cout
				<< "failed to parse arg in scad vectorised add";
		}
		operands.push_back(arg);
	}

	auto type = operands[0].getType();

	return builder.create<Operation>(
		location, type, operands[0], operands[1]
	);
}

mlir::Value SCADMIRLowering::scad_index(FFIHIRFunctionCall fc) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "index_op", 100, 100);

	// Codegen the operands first.
	auto array = codegen(fc.params[0]);
	auto index = codegen(fc.params[1]);

	SmallVector<mlir::Value, 4> indicies;
	indicies.push_back(index);

	return builder.create<mlir::memref::LoadOp>(location, array, indicies);
}

mlir::LogicalResult SCADMIRLowering::scad_prefetch_read(FFIHIRFunctionCall fc) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "index_op", 100, 100);

	// Codegen the operands first.
	auto memref = codegen(fc.params[0]);
	auto start = codegen(fc.params[1]);
	auto end = codegen(fc.params[2]);

	SmallVector<mlir::Value, 4> indicies;
	indicies.push_back(start);
	// indicies.push_back(end);

	builder.create<mlir::memref::PrefetchOp>(location, memref, indicies, false, 3, true);
	return mlir::success();
}

mlir::LogicalResult SCADMIRLowering::scad_prefetch_write(FFIHIRFunctionCall fc) {
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, "index_op", 100, 100);

	// Codegen the operands first.
	auto memref = codegen(fc.params[0]);
	auto start = codegen(fc.params[1]);
	auto end = codegen(fc.params[2]);

	SmallVector<mlir::Value, 4> indicies;
	indicies.push_back(start);
	// indicies.push_back(end);

	builder.create<mlir::memref::PrefetchOp>(location, memref, indicies, true, 3, true);
	return mlir::success();
}


mlir::scad::FuncOp
SCADMIRLowering::proto_gen(FFIHIRFunctionDecl ffd, FFIType function_type) {
	std::string name = std::string(ffd.name.data, ffd.name.size);
	mlir::Location location =
		mlir::FileLineColLoc::get(&context, name + "PROTO", 100, 100);

	llvm::SmallVector<mlir::Type, 4> arg_types;
	for (size_t i = 0; i < ffd.arg_len; i++) {
		auto type = get_type_for(function_type.apps[i]);
		arg_types.push_back(type);
	}

	auto type = builder.getFunctionType(arg_types, std::nullopt);

	if (name != "main") {
		function_results[name] =
			get_type_for(function_type.apps[function_type.size - 1]
			);
	}

	return builder.create<mlir::scad::FuncOp>(location, name, type);
}

mlir::scad::FuncOp SCADMIRLowering::scad_func(FFIHIRFunctionDecl decl) {
	std::string name = std::string(decl.name.data, decl.name.size);

	mlir::Location location =
		mlir::FileLineColLoc::get(&context, name + " Decl", 100, 100);
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
		returnOp =
			llvm::dyn_cast<mlir::scad::ReturnOp>(entryBlock.back());
	if (!returnOp) {
		builder.create<mlir::scad::ReturnOp>(location);
	} else if (returnOp.hasOperand() && name != "main") {
		// Otherwise, if this return operation has an operand then add a result to
		// the function.
		auto rettype = get_type_for(type.apps[type.size - 1]);

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

mlir::LogicalResult SCADMIRLowering::scad_return(FFIHIRReturn ret) {
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

mlir::LogicalResult SCADMIRLowering::scad_yield(FFIHIRYield yld) {
	mlir::Location location = mlir::FileLineColLoc::get(
		&context, std::string("YIELD STATEMENT!!!"), 100, 100
	);

	// std::string refer = std::string(
	// 	yld.res.value.variable_reference.name.data,
	// 	yld.res.value.variable_reference.name.size
	// );
	auto ret_val = codegen(yld.res);

	builder.create<mlir::scad::YieldOp>(location, ArrayRef(ret_val));
	return mlir::success();
}