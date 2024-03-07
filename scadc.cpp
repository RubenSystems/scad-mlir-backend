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

#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cstdlib>


#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

int main(int argc, const char * argv[]) {
	mlir::registerAsmPrinterCLOptions();
	mlir::registerMLIRContextCLOptions();
	mlir::registerPassManagerCLOptions();

	void * query_engine = nullptr;
	auto res = compile(argv[1], &query_engine);
	if (res.compiled == false) {
		return -1;
	}
	struct FFIHIRExpr x = res.program.prog;

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
	context.getOrLoadDialect<mlir::vector::VectorDialect>();
	context.getOrLoadDialect<mlir::scf::SCFDialect>();


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
	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed lower from affine\n";
		return -1;
	}
	owned_mod->dump();

	{
		mlir::OpPassManager & optPM = pm.nest<mlir::func::FuncOp>();
		optPM.addPass(mlir::createCanonicalizerPass());
		optPM.addPass(mlir::createCSEPass());
		optPM.addPass(mlir::affine::createLoopFusionPass());
		optPM.addPass(mlir::affine::createAffineScalarReplacementPass()
		);
		optPM.addPass(mlir::affine::createLoopUnrollPass());
		// optPM.addPass(mlir::affine::createLoopTilingPass());

		optPM.addPass(mlir::affine::createAffineVectorize());
		optPM.addPass(mlir::affine::createSimplifyAffineStructuresPass()
		);
		optPM.addPass(mlir::createLowerAffinePass()
		);
	}

	std::cout << "\n\n\n";

	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed lower from affine\n";
		return -1;
	}
	owned_mod->dump();
	std::cout << "\n\n\nnewmod:" << std::endl;
	pm.addPass(mlir::createForLoopPeelingPass());
	pm.addPass(mlir::createInlinerPass());
	pm.addPass(mlir::createArithToLLVMConversionPass());
	pm.addPass(mlir::createConvertSCFToOpenMPPass());
	pm.addPass(mlir::createConvertSCFToCFPass());

	pm.addPass(mlir::memref::createExpandOpsPass());
	pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
	pm.addPass(mlir::createConvertVectorToLLVMPass());
	pm.addPass(mlir::createConvertFuncToLLVMPass());
	pm.addPass(mlir::createConvertControlFlowToLLVMPass());

	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed to lower to llvm\n";
		return -1;
	}
	owned_mod->dump();

	pm.addPass(mlir::createCanonicalizerPass());
	pm.addPass(mlir::createCSEPass());
	pm.addPass(mlir::createMem2Reg());
	pm.addPass(mlir::createRemoveDeadValuesPass());
	pm.addPass(mlir::scad::createLowerToLLVMPass());
	pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
	pm.addPass(mlir::createCSEPass());
	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed to clean llvm\n";
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
	llvm::raw_fd_ostream ll_dest("tmp.ll", ec, llvm::sys::fs::OF_None);
	ll_dest << *llvmModule;
	llvm::errs() << *llvmModule << "\n";
	const char * llc_path_env_name = "TRUNK_LLC_PATH";
	char * value = std::getenv(llc_path_env_name);

	std::string command =
		std::string(value) + std::string(
			" -filetype=obj -o "
		) +
		argv[2] + std::string(" tmp.ll; rm tmp.ll");
	int result = system(command.data());

	return 0;
}
