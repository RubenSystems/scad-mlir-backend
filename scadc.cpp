#include <iostream>
#include <mlir_generator.h>
#include <passes.h>
#include <ast.h>

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, const char * argv[]) {
	mlir::registerAsmPrinterCLOptions();
	mlir::registerMLIRContextCLOptions();
	mlir::registerPassManagerCLOptions();

	void * query_engine = nullptr;
	FFIHIRExpr x = compile(argv[1], &query_engine);



	mlir::DialectRegistry registry;
	mlir::func::registerAllExtensions(registry);

	mlir::MLIRContext context(registry);

	mlir::registerOpenMPDialectTranslation(registry);
	mlir::registerOpenMPDialectTranslation(context);

	mlir::OpBuilder builder(&context);
	mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());
	mlir::OwningOpRef<mlir::ModuleOp> owned_mod = mod;
	context.getOrLoadDialect<mlir::scad::SCADDialect>();
	context.getOrLoadDialect<mlir::arith::ArithDialect>();
	context.getOrLoadDialect<mlir::memref::MemRefDialect>();
	context.getOrLoadDialect<mlir::affine::AffineDialect>();

	SCADMIRLowering scad_lowerer(context, builder, mod, query_engine);
	scad_lowerer.codegen(x);

	// auto function_a = generate_mlir_func_v2(builder, mod, context);
	// auto function = generate_mlir_func(builder, mod, context);

	mlir::PassManager pm(owned_mod.get()->getName());

	if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
		std::cout << "failed to apply cl options\n";
		return -1;
	}

	std::cout << "HIERE!!!" << std::endl;
	owned_mod->dump();
	std::cout << "HIERE!!!" << std::endl;

	{
		mlir::OpPassManager & optPM = pm.nest<mlir::scad::FuncOp>();
		optPM.addPass(mlir::createCanonicalizerPass());
		optPM.addPass(mlir::createCSEPass());


	}

	pm.addPass(mlir::scad::createLowerToAffinePass());


	{
		mlir::OpPassManager & optPM = pm.nest<mlir::func::FuncOp>();
		optPM.addPass(mlir::createLowerAffinePass());
		optPM.addPass(mlir::createCanonicalizerPass());
		optPM.addPass(mlir::createCSEPass());
		optPM.addPass(mlir::affine::createLoopFusionPass());
		optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
		optPM.addPass(mlir::affine::createLoopUnrollPass());
		optPM.addPass(mlir::affine::createAffineVectorize());
	}


	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed to run passes\n";
		return -1;
	}
	std::cout << "\n\n\n";
	owned_mod->dump();
	std::cout << "\n\n\n";
	pm.addPass(mlir::createConvertSCFToOpenMPPass());
    // pm.addPass(mlir::createConvertFuncToLLVMPass());
    // pm.addPass(mlir::createConvertMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertSCFToCFPass());
	pm.addPass(mlir::createConvertOpenMPToLLVMPass());
    // pm.addPass(mlir::createConvertIndexToLLVMPass());
    // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    // pm.addPass(mlir::memref::createExpandStridedMetadataPass());

    // pm.addPass(mlir::createConvertMathToLLVMPass());

    // pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    // pm.addPass(mlir::createConvertMathToLibmPass());

	pm.addPass(mlir::createCanonicalizerPass());
	pm.addPass(mlir::createCSEPass());
	pm.addPass(mlir::createMem2Reg());
	pm.addPass(mlir::createRemoveDeadValuesPass());

	pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed to run passes\n";
		return -1;
	}
	owned_mod->dump();

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	mlir::registerBuiltinDialectTranslation(*mod->getContext());
	mlir::registerLLVMDialectTranslation(*mod->getContext());

	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
	if (!llvmModule) {
		llvm::errs() << "Failed to convert to llvm engine\n";
		return -1;
	}


	auto optPipeline = mlir::makeOptimizingTransformer(
		/*optLevel=*/3,
		/*sizeLevel=*/0,
		/*targetMachine=*/nullptr
	);
	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
		return -1;
	}

	std::error_code ec;
	llvm::raw_fd_ostream ll_dest("output.ll", ec, llvm::sys::fs::OF_None);
	ll_dest << *llvmModule;
	llvm::errs() << *llvmModule << "\n";





	// llvm::errs() << "\n\n====EXECUTING====\n\n";
	// mlir::ExecutionEngineOptions engineOptions;
	// engineOptions.transformer = optPipeline;
	// auto maybeEngine = mlir::ExecutionEngine::create(mod, engineOptions);
	// assert(maybeEngine && "failed to construct an execution engine");
	// auto & engine = maybeEngine.get();
	// engine->dumpToObjectFile("output.o");

	// // Invoke the JIT-compiled function.
	// auto invocationResult = engine->invokePacked("main");
	// if (invocationResult) {
	// 	llvm::errs() << "JIT invocation failed\n";
	// 	return -1;
	// }
}