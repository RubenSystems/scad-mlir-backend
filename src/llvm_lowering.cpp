#include "Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>

using namespace mlir;

namespace {
	struct SCADToLLVMLoweringPass : public PassWrapper<
						SCADToLLVMLoweringPass,
						OperationPass<ModuleOp> > {
		MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
			SCADToLLVMLoweringPass
		)

		void getDependentDialects(DialectRegistry & registry
		) const override {
			registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
		}
		void runOnOperation() final;
	};
} // namespace

void SCADToLLVMLoweringPass::runOnOperation() {
	// The first thing to define is the conversion target. This will define the
	// final target for this lowering. For this lowering, we are only targeting
	// the LLVM dialect.
	LLVMConversionTarget target(getContext());
	target.addLegalOp<ModuleOp>();

	// During this lowering, we will also be lowering the MemRef types, that are
	// currently being operated on, to a representation in LLVM. To perform this
	// conversion we use a TypeConverter as part of the lowering. This converter
	// details how one type maps to another. This is necessary now that we will be
	// doing more complicated lowerings, involving loop region arguments.
	LLVMTypeConverter typeConverter(&getContext());

	// Now that the conversion target has been defined, we need to provide the
	// patterns used for lowering. At this point of the compilation process, we
	// have a combination of `toy`, `affine`, and `std` operations. Luckily, there
	// are already exists a set of patterns to transform `affine` and `std`
	// dialects. These patterns lowering in multiple stages, relying on transitive
	// lowerings. Transitive lowering, or A->B->C lowering, is when multiple
	// patterns must be applied to fully transform an illegal operation into a
	// set of legal ones.
	RewritePatternSet patterns(&getContext());
	populateAffineToStdConversionPatterns(patterns);
	populateSCFToControlFlowConversionPatterns(patterns);
	mlir::arith::populateArithToLLVMConversionPatterns(
		typeConverter, patterns
	);
	populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
	cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
	populateFuncToLLVMConversionPatterns(typeConverter, patterns);

	// The only remaining operation to lower from the `toy` dialect, is the
	// PrintOp.
	//   patterns.add<PrintOpLowering>(&getContext());

	// We want to completely lower to LLVM, so we use a `FullConversion`. This
	// ensures that only legal operations will remain after the conversion.
	auto module = getOperation();
	if (failed(applyFullConversion(module, target, std::move(patterns))))
		signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::scad::createLowerToLLVMPass() {
	return std::make_unique<SCADToLLVMLoweringPass>();
}