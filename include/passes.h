#pragma once 
#include <memory>


namespace mlir {
	class Pass;

	namespace scad {

		std::unique_ptr<mlir::Pass> createLowerToAffinePass();

	} 
} 