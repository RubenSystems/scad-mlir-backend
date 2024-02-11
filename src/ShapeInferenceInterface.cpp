// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/IR/Operation.h"
// #include "mlir/IR/Types.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Support/LLVM.h"
// #include "mlir/Support/TypeID.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/ADT/SmallPtrSet.h"
// #include "llvm/Support/Casting.h"
// #include "llvm/Support/Debug.h"
// #include "llvm/Support/raw_ostream.h"
// #include <memory>

// #include "Dialect.h"
// #include "passes.h"
// #include "ShapeInferenceInterface.h"

// using namespace mlir;
// using namespace scad;

// #define DEBUG_TYPE "shape-inference"
// #include "ShapeInferenceOpInterfaces.cpp.inc"


// namespace {

// 	struct ShapeInferencePass : public mlir::PassWrapper<
// 					    ShapeInferencePass,
// 					    OperationPass<scad::FuncOp> > {
// 		MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

// 		void runOnOperation() override {
// 			mlir::scad::FuncOp f = getOperation();

// 			llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
// 			f.walk([&](mlir::Operation * op) {
// 				if (returnsDynamicShape(op))
// 					opWorklist.insert(op);
// 			});

// 			// Iterate on the operations in the worklist until all operations have been
// 			// inferred or no change happened (fix point).
// 			while (!opWorklist.empty()) {
// 				// Find the next operation ready for inference, that is an operation
// 				// with all operands already resolved (non-generic).
// 				auto nextop = llvm::find_if(
// 					opWorklist, allOperandsInferred
// 				);
// 				if (nextop == opWorklist.end())
// 					break;

// 				Operation * op = *nextop;
// 				opWorklist.erase(op);

// 				// Ask the operation to infer its output shapes.
// 				LLVM_DEBUG(
// 					llvm::dbgs() << "Inferring shape for: "
// 						     << *op << "\n"
// 				);
// 				if (auto shapeOp =
// 					    dyn_cast<ShapeInference>(op)) {
// 					shapeOp.inferShapes();
// 				} else {
// 					op->emitError(
// 						"unable to infer shape of operation without shape "
// 						"inference interface"
// 					);
// 					return signalPassFailure();
// 				}
// 			}

// 			// If the operation worklist isn't empty, this indicates a failure.
// 			if (!opWorklist.empty()) {
// 				f.emitError("Shape inference failed, ")
// 					<< opWorklist.size()
// 					<< " operations couldn't be inferred\n";
// 				signalPassFailure();
// 			}
// 		}

// 		/// A utility method that returns if the given operation has all of its
// 		/// operands inferred.
// 		static bool allOperandsInferred(Operation * op) {
// 			return llvm::all_of(
// 				op->getOperandTypes(),
// 				[](Type operandType) {
// 					return llvm::isa<RankedTensorType>(
// 						operandType
// 					);
// 				}
// 			);
// 		}

// 		/// A utility method that returns if the given operation has a dynamically
// 		/// shaped result.
// 		static bool returnsDynamicShape(Operation * op) {
// 			return llvm::any_of(
// 				op->getResultTypes(),
// 				[](Type resultType) {
// 					return !llvm::isa<RankedTensorType>(
// 						resultType
// 					);
// 				}
// 			);
// 		}
// 	};
// } // namespace

// /// Create a Shape Inference pass.
// std::unique_ptr<mlir::Pass> mlir::scad::createShapeInferencePass() {
// 	return std::make_unique<ShapeInferencePass>();
// }