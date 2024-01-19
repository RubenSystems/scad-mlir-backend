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
#include "passes.h"

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
static MemRefType convertTensorToMemRef(RankedTensorType type) {
	return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value
insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter & rewriter) {
	auto alloc = rewriter.create<memref::AllocOp>(loc, type);

	// Make sure to allocate at the beginning of the block.
	auto * parentBlock = alloc->getBlock();
	alloc->moveBefore(&parentBlock->front());

	// Make sure to deallocate this alloc at the end of the block. This is fine
	// as toy functions have no control flow.
	auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
	dealloc->moveBefore(&parentBlock->back());
	return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<
	Value(OpBuilder & rewriter, ValueRange memRefOperands, ValueRange loopIvs
	)>;

static void lowerOpToLoops(
	Operation * op,
	ValueRange operands,
	PatternRewriter & rewriter,
	LoopIterationFn processIteration
) {
	auto tensorType =
		llvm::cast<RankedTensorType>((*op->result_type_begin()));
	auto loc = op->getLoc();

	// Insert an allocation and deallocation for the result of this operation.
	auto memRefType = convertTensorToMemRef(tensorType);
	auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

	// Create a nest of affine loops, with one loop per dimension of the shape.
	// The buildAffineLoopNest function takes a callback that is used to construct
	// the body of the innermost loop given a builder, a location and a range of
	// loop induction variables.
	SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
	SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
	affine::buildAffineLoopNest(
		rewriter,
		loc,
		lowerBounds,
		tensorType.getShape(),
		steps,
		[&](OpBuilder & nestedBuilder, Location loc, ValueRange ivs) {
			// Call the processing function with the rewriter, the memref operands,
			// and the loop induction variables. This function will return the value
			// to store at the current index.
			Value valueToStore =
				processIteration(nestedBuilder, operands, ivs);
			nestedBuilder.create<affine::AffineStoreOp>(
				loc, valueToStore, alloc, ivs
			);
		}
	);

	// Replace this operation with the generated alloc.
	rewriter.replaceOp(op, alloc);
}

struct VectorOpLowering : public OpRewritePattern<scad::VectorOp> {
	using OpRewritePattern<scad::VectorOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		scad::VectorOp op,
		PatternRewriter & rewriter
	) const final {
		DenseElementsAttr constantValue = op.getValue();
		Location loc = op.getLoc();

		// When lowering the constant operation, we allocate and assign the constant
		// values to a corresponding memref allocation.
		auto tensorType = llvm::cast<RankedTensorType>(op.getType());
		auto memRefType = convertTensorToMemRef(tensorType);
		auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

		// We will be generating constant indices up-to the largest dimension.
		// Create these constants up-front to avoid large amounts of redundant
		// operations.
		auto valueShape = memRefType.getShape();
		SmallVector<Value, 8> constantIndices;

		if (!valueShape.empty()) {
			for (auto i : llvm::seq<int64_t>(
				     0,
				     *std::max_element(
					     valueShape.begin(),
					     valueShape.end()
				     )
			     ))
				constantIndices.push_back(
					rewriter.create<arith::ConstantIndexOp>(
						loc, i
					)
				);
		} else {
			// This is the case of a tensor of rank 0.
			constantIndices.push_back(
				rewriter.create<arith::ConstantIndexOp>(loc, 0)
			);
		}

		// The constant operation represents a multi-dimensional constant, so we
		// will need to generate a store for each of the elements. The following
		// functor recursively walks the dimensions of the constant shape,
		// generating a store when the recursion hits the base case.
		SmallVector<Value, 2> indices;
		auto valueIt = constantValue.value_begin<IntegerAttr>();
		std::function<void(uint64_t)> storeElements = [&](uint64_t dimension
							      ) {
			// The last dimension is the base case of the recursion, at this point
			// we store the element at the given index.
			if (dimension == valueShape.size()) {
				rewriter.create<affine::AffineStoreOp>(
					loc,
					rewriter.create<arith::ConstantOp>(
						loc, *valueIt++
					),
					alloc,
					llvm::ArrayRef(indices)
				);
				return;
			}

			// Otherwise, iterate over the current dimension and add the indices to
			// the list.
			for (uint64_t i = 0, e = valueShape[dimension]; i != e;
			     ++i) {
				indices.push_back(constantIndices[i]);
				storeElements(dimension + 1);
				indices.pop_back();
			}
		};

		// Start the element storing recursion from the first dimension.
		storeElements(/*dimension=*/0);

		// Replace this operation with the generated alloc.
		rewriter.replaceOp(op, alloc);
		return success();
	}
};

struct FuncOpLowering : public OpConversionPattern<scad::FuncOp> {
	using OpConversionPattern<scad::FuncOp>::OpConversionPattern;

	LogicalResult matchAndRewrite(
		scad::FuncOp op,
		OpAdaptor adaptor,
		ConversionPatternRewriter & rewriter
	) const final {
		// We only lower the main function as we expect that all other functions
		// have been inlined.
		if (op.getName() != "main")
			return failure();

		// Verify that the given main has no inputs and results.
		if (op.getNumArguments() ||
		    op.getFunctionType().getNumResults()) {
			return rewriter.notifyMatchFailure(op, [](Diagnostic & diag) {
				diag << "expected 'main' to have 0 inputs and 0 results";
			});
		}

		// Create a new non-toy function, with the same region.
		auto func = rewriter.create<mlir::func::FuncOp>(
			op.getLoc(), op.getName(), op.getFunctionType()
		);
		rewriter.inlineRegionBefore(
			op.getRegion(), func.getBody(), func.end()
		);
		rewriter.eraseOp(op);
		return success();
	}
};

struct ReturnOpLowering : public OpRewritePattern<scad::ReturnOp> {
	using OpRewritePattern<scad::ReturnOp>::OpRewritePattern;

	LogicalResult matchAndRewrite(
		scad::ReturnOp op,
		PatternRewriter & rewriter
	) const final {
		// During this lowering, we expect that all function calls have been
		// inlined.
		if (op.hasOperand())
			return failure();

		rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
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
	// The first thing to define is the conversion target. This will define the
	// final target for this lowering.
	std::cout << "HI!\n";
	ConversionTarget target(getContext());

	// We define the specific operations, or dialects, that are legal targets for
	// this lowering. In our case, we are lowering to a combination of the
	// `Affine`, `Arith`, `Func`, and `MemRef` dialects.
	target.addLegalDialect<
		affine::AffineDialect,
		BuiltinDialect,
		arith::ArithDialect,
		func::FuncDialect,
		memref::MemRefDialect>();

	// We also define the Toy dialect as Illegal so that the conversion will fail
	// if any of these operations are *not* converted. Given that we actually want
	// a partial lowering, we explicitly mark the Toy operations that don't want
	// to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
	// to be updated though (as we convert from TensorType to MemRefType), so we
	// only treat it as `legal` if its operands are legal.
	// target.addIllegalDialect<scad::SCADDialect>();
	// target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
	//   return llvm::none_of(op->getOperandTypes(),
	//                        [](Type type) { return llvm::isa<TensorType>(type);
	//                        });
	// });

	// Now that the conversion target has been defined, we just need to provide
	// the set of patterns that will lower the Toy operations.
	RewritePatternSet patterns(&getContext());
	patterns.add<VectorOpLowering, FuncOpLowering, ReturnOpLowering>(&getContext());

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our `illegal`
	// operations were not converted successfully.
	if (failed(applyPartialConversion(
		    getOperation(), target, std::move(patterns)
	    )))
		signalPassFailure();
}

std::unique_ptr<Pass> mlir::scad::createLowerToAffinePass() {
	return std::make_unique<SCADToAffineLoweringPass>();
}
