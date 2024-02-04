#include <iostream>
#include <mlir_generator.h>
#include <passes.h>
#include <ast.h>

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
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

extern "C" {
	FFIHIRExpr compile();
}

int dumpLLVMIR(mlir::ModuleOp module) {
	// Register the translation to LLVM IR with the MLIR context.
	mlir::registerBuiltinDialectTranslation(*module->getContext());
	mlir::registerLLVMDialectTranslation(*module->getContext());

	// Convert the module to LLVM IR in a new LLVM IR context.
	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return -1;
	}

	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Configure the LLVM Module
	auto tmBuilderOrError =
		llvm::orc::JITTargetMachineBuilder::detectHost();
	if (!tmBuilderOrError) {
		llvm::errs() << "Could not create JITTargetMachineBuilder\n";
		return -1;
	}

	auto tmOrError = tmBuilderOrError->createTargetMachine();
	if (!tmOrError) {
		llvm::errs() << "Could not create TargetMachine\n";
		return -1;
	}
	mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
		llvmModule.get(), tmOrError.get().get()
	);

	/// Optionally run an optimization pipeline over the llvm module.
	auto optPipeline = mlir::makeOptimizingTransformer(
		/*optLevel=*/3,
		/*sizeLevel=*/0,
		/*targetMachine=*/nullptr
	);
	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
		return -1;
	}
	llvm::errs() << *llvmModule << "\n";
	return 0;
}

int main() {
	mlir::registerAsmPrinterCLOptions();
	mlir::registerMLIRContextCLOptions();
	mlir::registerPassManagerCLOptions();

	FFIHIRExpr x = compile();

	mlir::DialectRegistry registry;
	mlir::func::registerAllExtensions(registry);

	mlir::MLIRContext context(registry);
	mlir::OpBuilder builder(&context);
	mlir::ModuleOp mod = mlir::ModuleOp::create(builder.getUnknownLoc());
	mlir::OwningOpRef<mlir::ModuleOp> owned_mod = mod;
	context.getOrLoadDialect<mlir::scad::SCADDialect>();

	SCADMIRLowering scad_lowerer(context, builder, mod);

	scad_lowerer.codegen(x);

	// auto function_a = generate_mlir_func_v2(builder, mod, context);
	// auto function = generate_mlir_func(builder, mod, context);

	mlir::PassManager pm(owned_mod.get()->getName());

	if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
		std::cout << "failed to apply cl options\n";
		return -1;
	}

	owned_mod->dump();
	pm.addPass(mlir::scad::createLowerToAffinePass());
	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed to run passes\n";
		return -1;
	}
	std::cout << "\n\n\n";
	owned_mod->dump();
	std::cout << "\n\n\n";
	pm.addPass(mlir::scad::createLowerToLLVMPass());
	if (mlir::failed(pm.run(*owned_mod))) {
		std::cout << "failed to run passes\n";
		return -1;
	}
	owned_mod->dump();

	mlir::registerBuiltinDialectTranslation(*mod->getContext());
	mlir::registerLLVMDialectTranslation(*mod->getContext());

	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(mod, llvmContext);
	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return -1;
	}

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	auto tmBuilderOrError =
		llvm::orc::JITTargetMachineBuilder::detectHost();
	if (!tmBuilderOrError) {
		llvm::errs() << "Could not create JITTargetMachineBuilder\n";
		return -1;
	}

	auto tmOrError = tmBuilderOrError->createTargetMachine();
	if (!tmOrError) {
		llvm::errs() << "Could not create TargetMachine\n";
		return -1;
	}
	mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
		llvmModule.get(), tmOrError.get().get()
	);

	/// Optionally run an optimization pipeline over the llvm module.
	auto optPipeline = mlir::makeOptimizingTransformer(
		/*optLevel=*/3,
		/*sizeLevel=*/0,
		/*targetMachine=*/nullptr
	);
	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
		return -1;
	}
	llvm::errs() << *llvmModule << "\n";
}