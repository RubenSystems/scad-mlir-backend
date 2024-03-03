#include <iostream>

#include "Dialect.h"
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
#include "passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"

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
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

// Code borrowed from mlir docs

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convert_tensor_type_to_memref_type(RankedTensorType type) {
	return MemRefType::get(type.getShape(), type.getElementType());
}

// using AddOpLowering = BinaryOpLowering<scad::AddOp, arith::AddIOp>;
struct AddOpLowering : public OpConversionPattern<scad::AddOp> {
	using OpConversionPattern<scad::AddOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::AddOp op,
		OpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		rewriter.replaceOpWithNewOp<arith::AddIOp>(
			op, adaptor.getLhs(), adaptor.getRhs()
		);

		return success();
	}
};

struct FuncOpLowering : public OpConversionPattern<scad::FuncOp> {
	using OpConversionPattern<scad::FuncOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::FuncOp op,
		mlir::scad::FuncOpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		if (op.getName() == "main") {
			auto func = rewriter.create<mlir::func::FuncOp>(
				op.getLoc(), op.getName(), op.getFunctionType()
			);
			rewriter.inlineRegionBefore(
				op.getRegion(), func.getBody(), func.end()
			);

		} else {
			// Assumes funcitons have more then one restype
			auto res_type =
				adaptor.getFunctionType().getResults()[0];

			auto function_type = rewriter.getFunctionType(
				op.getFunctionType().getInputs(), res_type
			);

			auto func = rewriter.create<mlir::func::FuncOp>(
				op.getLoc(), op.getName(), function_type
			);

			mlir::Block & entry_block = op.front();
			auto entry_args = entry_block.getArguments();
			for (size_t i = 0; i < entry_args.size(); i++) {
				auto arg_type = function_type.getInput(i);
				entry_block.getArgument(i).setType(arg_type);
			}
			rewriter.inlineRegionBefore(
				op.getRegion(), func.getBody(), func.end()
			);
		}
		rewriter.eraseOp(op);
		return success();
	}
};

struct ConditionalOpLowering : public OpConversionPattern<scad::ConditionalOp> {
	using OpConversionPattern<scad::ConditionalOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::ConditionalOp op,
		mlir::scad::ConditionalOpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		auto conditional_set = mlir::IntegerSet::get(
			1, 0, rewriter.getAffineDimExpr(0) - 10, { false }

		);

		mlir::Type type = mlir::IntegerType::get(
			rewriter.getContext(),
			32,
			mlir::IntegerType::SignednessSemantics::Signless
		);

		mlir::Value constant =
			rewriter.create<mlir::arith::ConstantIndexOp>(
				rewriter.getUnknownLoc(), 100
			);

		auto cond = rewriter.replaceOpWithNewOp<affine::AffineIfOp>(
			op,
			MemRefType::get({ 2 }, rewriter.getI32Type()),
			conditional_set,
			constant,
			true
		);

		rewriter.inlineBlockBefore(
			&adaptor.getIfArm().front(),
			&cond.getThenRegion().front(),
			cond.getThenRegion().back().end()
		);

		rewriter.inlineBlockBefore(
			&adaptor.getElseArm().front(),
			&cond.getElseRegion().front(),
			cond.getElseRegion().back().end()
		);

		// auto block = cond.getThenRegion().front();

		return success();
	}
};

// struct PrintOpLowering : public OpConversionPattern<scad::PrintOp> {
// 	using OpConversionPattern<scad::PrintOp>::OpConversionPattern;

// 	LogicalResult matchAndRewrite(
// 		scad::PrintOp op,
// 		OpAdaptor adaptor,
// 		ConversionPatternRewriter & rewriter
// 	) const final {
// 		rewriter.modifyOpInPlace(op, [&] {
// 			op->setOperands(adaptor.getOperands());
// 		});
// 		return success();
// 	}
// };

struct ReturnOpLowering : public OpConversionPattern<scad::ReturnOp> {
	using OpConversionPattern<scad::ReturnOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::ReturnOp op,
		OpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		rewriter.replaceOpWithNewOp<func::ReturnOp>(
			op, adaptor.getOperands()
		);

		return success();
	}
};

struct YieldOpLowering : public OpConversionPattern<scad::YieldOp> {
	using OpConversionPattern<scad::YieldOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::YieldOp op,
		OpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(
			op, adaptor.getOperands()
		);

		return success();
	}
};

// struct IndexOpLowering : public OpConversionPattern<scad::IndexOp> {
// 	using OpConversionPattern<scad::IndexOp>::OpConversionPattern;

// 	LogicalResult matchAndRewrite(
// 		scad::IndexOp op,
// 		OpAdaptor adaptor,
// 		ConversionPatternRewriter & rewriter
// 	) const final {
// 		auto memRefType = op.getValue().getType();
// 		SmallVector<Value, 8> constant_indicies;
// 		// constant_indicies.push_back(
// 		// 	rewriter.create<arith::ConstantIndexOp>(
// 		// 		op.getLoc(), op.getODSOperands(1)
// 		// 	)
// 		// );
// 		auto memRef = rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
// 			op, adaptor.getValue(), op.getODSOperands(1)
// 		);

// 		return success();
// 	}
// };

struct CallOpLowering : public OpConversionPattern<mlir::scad::GenericCallOp> {
	using OpConversionPattern<mlir::scad::GenericCallOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::GenericCallOp op,
		mlir::scad::GenericCallOpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		StringRef callee = op.getCalleeAttrName();

		auto inputs = adaptor.getInputs();

		auto call_op = rewriter.replaceOpWithNewOp<func::CallOp>(
			op,
			callee,
			*op->result_type_begin(),
			adaptor.getOperands()
		);
		call_op->setAttrs(op->getAttrs());

		// rewriter.eraseOp(op);

		return success();
	}
};

namespace {
	struct SCADToAffineLoweringPass : public PassWrapper<
						  SCADToAffineLoweringPass,
						  OperationPass<ModuleOp> > {
		MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
			SCADToAffineLoweringPass
		)

		void getDependentDialects(DialectRegistry & registry
		) const override {
			registry
				.insert<affine::AffineDialect,
					func::FuncDialect,
					memref::MemRefDialect>();
		}
		void runOnOperation() final;
	};
} // namespace

void SCADToAffineLoweringPass::runOnOperation() {
	ConversionTarget target(getContext());

	target.addLegalDialect<
		affine::AffineDialect,
		BuiltinDialect,
		arith::ArithDialect,
		func::FuncDialect,
		mlir::vector::VectorDialect,
		mlir::arm_sve::ArmSVEDialect,
		memref::MemRefDialect>();

	target.addDynamicallyLegalOp<scad::PrintOp>([](scad::PrintOp op) {
		return llvm::none_of(op->getOperandTypes(), [](Type type) {
			return llvm::isa<TensorType>(type);
		});
	});

	RewritePatternSet patterns(&getContext());
	patterns
		.add<FuncOpLowering,
		     ReturnOpLowering,
		     CallOpLowering,
		     AddOpLowering,
		     ConditionalOpLowering,
		     YieldOpLowering>(&getContext());

	if (failed(applyPartialConversion(
		    getOperation(), target, std::move(patterns)
	    )))
		signalPassFailure();
}

std::unique_ptr<Pass> mlir::scad::createLowerToAffinePass() {
	return std::make_unique<SCADToAffineLoweringPass>();
}