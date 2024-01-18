#include <iostream> 
#include <mlir_generator.h>
#include <passes.h>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

int main () {
	
	mlir::MLIRContext context;
	context.getOrLoadDialect<mlir::scad::SCADDialect>();
	mlir::OwningOpRef<mlir::ModuleOp> mlir = generate_mlir(context);
	mlir::PassManager pm (mlir.get()->getName());
	mlir->dump();
	pm.addPass(mlir::scad::createLowerToAffinePass());
	mlir::failed(pm.run(*mlir));

	mlir->dump();
}