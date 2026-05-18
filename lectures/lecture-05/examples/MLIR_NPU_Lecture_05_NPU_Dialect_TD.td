// MLIR_NPU_Lecture_05_NPU_Dialect_TD.td
// Educational pseudo ODS/TableGen skeleton for an NPU kernel dialect.
// This is intended for architecture/design study, not as a drop-in build file.

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def NPUKernel_Dialect : Dialect {
  let name = "npu_kernel";
  let cppNamespace = "::npu::kernel";
  let summary = "Kernel-level NPU operations";
  let description = [{
    The npu_kernel dialect models tiled compute, DMA, barriers, scratchpad
    tiles, accumulator tiles and target-specific scheduling constraints.
  }];
}

//===----------------------------------------------------------------------===//
// Common op base
//===----------------------------------------------------------------------===//

class NPUKernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPUKernel_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Types / attributes - pseudo placeholders
//===----------------------------------------------------------------------===//

// !npu_kernel.sram_tile<16x32xi8>
// !npu_kernel.acc_tile<16x16xi32>
// #npu_kernel.layout<NHK, bank_swizzle = 2>
// #npu_kernel.schedule<tile = [16,16,32], double_buffer = true>

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def NPUKernel_DmaLoadOp : NPUKernel_Op<"dma_load", []> {
  let summary = "Load a tile from DRAM/global memory to scratchpad SRAM";
  let arguments = (ins Index:$tile_m, Index:$tile_k,
                       I64Attr:$alignment,
                       I64Attr:$burst,
                       StrAttr:$memory_space);
  let results = (outs AnyType:$tile);
  let hasVerifier = 1;
}

def NPUKernel_BarrierOp : NPUKernel_Op<"barrier", []> {
  let summary = "Synchronize DMA and compute within a core or cluster";
  let arguments = (ins StrAttr:$scope);
  let results = (outs);
  let hasVerifier = 1;
}

def NPUKernel_MatMulOp : NPUKernel_Op<"matmul", []> {
  let summary = "Execute a tiled matrix multiply on the NPU compute array";
  let arguments = (ins AnyType:$lhs,
                       AnyType:$rhs,
                       I64ArrayAttr:$tile_shape,
                       StrAttr:$accumulator_type,
                       I64ArrayAttr:$array_shape);
  let results = (outs AnyType:$acc);
  let hasVerifier = 1;

  // Example verifier rules:
  // - lhs/rhs tile K dimensions must match.
  // - tile_shape must be legal for the array_shape.
  // - accumulator_type must be compatible with input element type.
  // - estimated SRAM footprint must fit in target capacity.
}

def NPUKernel_RequantReluOp : NPUKernel_Op<"requant_relu", []> {
  let summary = "Apply requantization and fused ReLU to accumulator tile";
  let arguments = (ins AnyType:$acc,
                       StrAttr:$scale_encoding,
                       StrAttr:$output_type);
  let results = (outs AnyType:$out_tile);
  let hasVerifier = 1;
}

def NPUKernel_DmaStoreOp : NPUKernel_Op<"dma_store", []> {
  let summary = "Store a tile from scratchpad SRAM to DRAM/global memory";
  let arguments = (ins AnyType:$tile,
                       Index:$tile_m,
                       Index:$tile_n,
                       I64Attr:$alignment,
                       StrAttr:$memory_space);
  let results = (outs);
  let hasVerifier = 1;
}
