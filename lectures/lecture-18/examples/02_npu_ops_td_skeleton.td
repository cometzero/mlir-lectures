//===- NPUOps.td - NPU dialect op definitions -----------*- tablegen -*-===//
// Educational skeleton. Adapt constraints and namespaces for a real compiler.

#ifndef NPU_OPS_TD
#define NPU_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "NPUTypes.td"

def NPU_Dialect : Dialect {
  let name = "npu";
  let cppNamespace = "::mlir::npu";
  let summary = "Target-specific NPU kernel and command dialect";
  let description = [{
    Captures NPU tile-level compute, DMA, and synchronization contracts after
    tiling, layout, memory-space, and quantization decisions have been made.
  }];
}

class NPU_Op<string mnemonic, list<Trait> traits = []> : Op<NPU_Dialect, mnemonic, traits>;

def NPU_MatmulOp : NPU_Op<"matmul", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Issue one native NPU matrix multiply tile command";
  let arguments = (ins NPU_TileBuffer:$lhs, NPU_TileBuffer:$rhs,
                       NPU_AccBuffer:$acc, NPU_TileShapeAttr:$tile,
                       NPU_LayoutAttr:$lhs_layout, NPU_LayoutAttr:$rhs_layout,
                       OptionalAttr<NPU_EpilogueAttr>:$epilogue);
  let results = (outs NPU_AccBuffer:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict `:`
    type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def NPU_DmaOp : NPU_Op<"dma", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Start an asynchronous DMA transfer";
  let arguments = (ins AnyType:$source, AnyType:$dest,
                       NPU_DmaDirectionAttr:$direction, I64Attr:$elements,
                       OptionalAttr<I64Attr>:$stride,
                       OptionalAttr<I64Attr>:$elements_per_stride);
  let results = (outs NPU_Token:$token);
  let assemblyFormat = "$source `->` $dest attr-dict `:` type($source) `to` type($dest)";
  let hasVerifier = 1;
}

def NPU_BarrierOp : NPU_Op<"barrier", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Wait for one or more NPU command tokens";
  let arguments = (ins Variadic<NPU_Token>:$tokens, NPU_BarrierScopeAttr:$scope);
  let results = (outs);
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
  let hasVerifier = 1;
}

#endif // NPU_OPS_TD
