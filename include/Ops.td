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

def VectorOp : Scad_Op<"vector", [Pure]> {

  let summary = "A quantity reperesented by both direction and magnitude";

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


def FuncOp : Scad_Op<"fn", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";


  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif // SCAD_OPS