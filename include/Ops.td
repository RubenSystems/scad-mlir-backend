#ifndef SCAD_OPS
#define SCAD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SCAD_Dialect : Dialect {
  let name = "scad";
  let cppNamespace = "::mlir::scad";
}


class Scad_Op<string mnemonic, list<Trait> traits = []> :
    Op<SCAD_Dialect, mnemonic, traits>;

def RegisterOp : Scad_Op<"register", [Pure]> {

  let summary = "register";

  let arguments = (ins I32ElementsAttr:$value);
  let results = (outs I32Tensor);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "DenseIntElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "int32_t":$value)>
  ];

}

#endif // SCAD_OPS