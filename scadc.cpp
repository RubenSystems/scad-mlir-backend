#include <iostream> 
#include <mlir_generator.h>


int main () {
	
	mlir::MLIRContext context;
	context.getOrLoadDialect<mlir::scad::SCADDialect>();

	mlir::OwningOpRef<mlir::ModuleOp> mlir = generate_mlir(context);
	mlir->dump();
}