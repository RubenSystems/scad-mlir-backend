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

	class PrintOpLowering : public ConversionPattern {
	    public:
		explicit PrintOpLowering(MLIRContext * context)
			: ConversionPattern(
				  scad::PrintOp::getOperationName(),
				  1,
				  context
			  ) {
		}

		LogicalResult matchAndRewrite(
			Operation * op,
			ArrayRef<Value> operands,
			ConversionPatternRewriter & rewriter
		) const override {
			auto * context = rewriter.getContext();
			// auto memRefType = llvm::cast<MemRefType>(
			// 	(*op->operand_type_begin())
			// );
			// auto memRefShape = memRefType.getShape();
			auto loc = op->getLoc();

			ModuleOp parentModule = op->getParentOfType<ModuleOp>();

			// Get a symbol reference to the printf function, inserting it if necessary.
			auto printfRef =
				getOrInsertPrintf(rewriter, parentModule);
			Value formatSpecifierCst = getOrCreateGlobalString(
				loc,
				rewriter,
				"frmt_spec",
				StringRef("%i \n\0", 4),
				parentModule
			);
			Value newLineCst = getOrCreateGlobalString(
				loc,
				rewriter,
				"nl",
				StringRef("\n\0", 2),
				parentModule
			);

			// Generate a call to printf for the current element of the loop.
			auto printOp = cast<scad::PrintOp>(op);
			// auto elementLoad = rewriter.create<memref::LoadOp>(
			// 	loc, printOp.getInput(), loopIvs
			// );
			rewriter.create<LLVM::CallOp>(
				loc,
				getPrintfType(context),
				printfRef,
				ArrayRef<Value>({ formatSpecifierCst,
						  operands[0] })
			);

			// Notify the rewriter that this operation has been removed.
			rewriter.eraseOp(op);
			return success();
		}

	    private:
		/// Create a function declaration for printf, the signature is:
		///   * `i32 (i8*, ...)`
		static LLVM::LLVMFunctionType
		getPrintfType(MLIRContext * context) {
			auto llvmI32Ty = IntegerType::get(context, 32);
			auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
			auto llvmFnType = LLVM::LLVMFunctionType::get(
				llvmI32Ty,
				llvmPtrTy,
				/*isVarArg=*/true
			);
			return llvmFnType;
		}

		/// Return a symbol reference to the printf function, inserting it into the
		/// module if necessary.
		static FlatSymbolRefAttr
		getOrInsertPrintf(PatternRewriter & rewriter, ModuleOp module) {
			auto * context = module.getContext();
			if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
				return SymbolRefAttr::get(context, "printf");

			// Insert the printf function into the body of the parent module.
			PatternRewriter::InsertionGuard insertGuard(rewriter);
			rewriter.setInsertionPointToStart(module.getBody());
			rewriter.create<LLVM::LLVMFuncOp>(
				module.getLoc(),
				"printf",
				getPrintfType(context)
			);
			return SymbolRefAttr::get(context, "printf");
		}

		/// Return a value representing an access into a global string with the given
		/// name, creating the string if necessary.
		static Value getOrCreateGlobalString(
			Location loc,
			OpBuilder & builder,
			StringRef name,
			StringRef value,
			ModuleOp module
		) {
			// Create the global at the entry of the module.
			LLVM::GlobalOp global;
			if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name)
			    )) {
				OpBuilder::InsertionGuard insertGuard(builder);
				builder.setInsertionPointToStart(module.getBody(
				));
				auto type = LLVM::LLVMArrayType::get(
					IntegerType::get(
						builder.getContext(), 8
					),
					value.size()
				);
				global = builder.create<LLVM::GlobalOp>(
					loc,
					type,
					/*isConstant=*/true,
					LLVM::Linkage::Internal,
					name,
					builder.getStringAttr(value),
					/*alignment=*/0
				);
			}

			// Get the pointer to the first character in the global string.
			Value globalPtr =
				builder.create<LLVM::AddressOfOp>(loc, global);
			Value cst0 = builder.create<LLVM::ConstantOp>(
				loc,
				builder.getI64Type(),
				builder.getIndexAttr(0)
			);
			return builder.create<LLVM::GEPOp>(
				loc,
				LLVM::LLVMPointerType::get(builder.getContext()
				),
				global.getType(),
				globalPtr,
				ArrayRef<Value>({ cst0, cst0 })
			);
		}
	};
}

struct SCADToLLVMLoweringPass
	: public PassWrapper<SCADToLLVMLoweringPass, OperationPass<ModuleOp> > {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCADToLLVMLoweringPass)

	void getDependentDialects(DialectRegistry & registry) const override {
		registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
	}
	void runOnOperation() final;
};

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

	patterns.add<PrintOpLowering>(&getContext());

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