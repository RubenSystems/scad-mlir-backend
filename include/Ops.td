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

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}


/*
=============
Conditional op
=============
*/
def ConditionalOp : Scad_Op<"cond"> {
  let summary = "if this then that else that";

  Type AnyValue = AnyTypeOf<[I1, I8, I16, I32, I64, Index, I1MemRef, I8MemRef, I16MemRef, I32MemRef, I64MemRef]>;
  let arguments = (ins I1:$condition);
  let results = (outs AnyValue);
  let regions = (region
    AnyRegion:$if_arm,
    AnyRegion:$else_arm
  );

  // let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$condition, "Type":$rettype)>
  ];
}


def ReturnOp : Scad_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";

  Type AnyValue = AnyTypeOf<[I1, I8, I16, I32, I64, Index, I1MemRef, I8MemRef, I16MemRef, I32MemRef, I64MemRef]>;

  let arguments = (ins Variadic<AnyValue>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

def YieldOp : Scad_Op<"yield", [Pure, HasParent<"ConditionalOp">,
                                 Terminator]> {
  let summary = "yield operation to provide a value for a conditional";

  Type AnyValue = AnyTypeOf<[I1, I8, I16, I32, I64, Index, I1MemRef, I8MemRef, I16MemRef, I32MemRef, I64MemRef]>;

  let arguments = (ins Variadic<AnyValue>:$input);

  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

def GenericCallOp : Scad_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";

    Type AnyValue = AnyTypeOf<[I1, I8, I16, I32, I64, Index, I1MemRef, I8MemRef, I16MemRef, I32MemRef, I64MemRef]>;

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyValue>:$inputs);

  let results = (outs AnyValue);

  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

def PrintOp : Scad_Op<"print"> {
  let arguments = (ins I32:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}


// def AddOp : Scad_Op<"add",
//     [Pure]> {
//   let summary = "element-wise addition operation";
//   Type AnyInt = AnyTypeOf<[I8, I16, I32, I64]>;
//   let arguments = (ins AnyInt:$lhs, AnyInt:$rhs);
//   let results = (outs AnyInt);

//   // Indicate that the operation has a custom parser and printer method.
//   // let hasCustomAssemblyFormat = 1;

//   // Allow building an AddOp with from the two input operands.
//   let builders = [
//     OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
//   ];
// }

// def DropOp : Scad_Op<"drop"> {
//   let summary = "deallocate";
//   let assemblyFormat = "$value attr-dict `:` type($value)";

//   let arguments = (ins I32MemRef:$value);

//   // let builders = [
//   //   OpBuilder<(ins "Value":$dropval)>
//   // ];
// }

// def IndexOp : Scad_Op<"index",
//     [Pure]> {
//   let summary = "Get element from tensor at index";

//   let arguments = (ins I32MemRef:$value, Index:$index);
//   let results = (outs I32);

//   let builders = [
//     OpBuilder<(ins "Value":$value, "Value":$index)>
//   ];

//   // let builders = [
//   //       OpBuilder<(ins "DenseIntElementsAttr":$value, "DenseIntElementsAttr":$index), [{
//   //       build($_builder, $_state, value.getType(), value, index.getType(), index);
//   //     }]>,

//   // ];
// }



#endif // SCAD_OPS
