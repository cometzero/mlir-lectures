//===- NPUTypes.td - NPU type/attribute skeleton ---------*- tablegen -*-===//
#ifndef NPU_TYPES_TD
#define NPU_TYPES_TD
include "mlir/IR/AttrTypeBase.td"

def NPU_TileBuffer : Type<CPred<"$_self.isa<::mlir::npu::TileBufferType>()">, "NPU SRAM tile buffer type">;
def NPU_AccBuffer  : Type<CPred<"$_self.isa<::mlir::npu::AccBufferType>()">, "NPU accumulator tile buffer type">;
def NPU_Token      : Type<CPred<"$_self.isa<::mlir::npu::TokenType>()">, "NPU async command token">;

def NPU_TileShapeAttr   : Attr<CPred<"$_self.isa<::mlir::npu::TileShapeAttr>()">, "native tile shape">;
def NPU_LayoutAttr      : Attr<CPred<"$_self.isa<::mlir::npu::LayoutAttr>()">, "tile layout">;
def NPU_DmaDirectionAttr: Attr<CPred<"$_self.isa<::mlir::npu::DmaDirectionAttr>()">, "DMA direction">;
def NPU_BarrierScopeAttr: Attr<CPred<"$_self.isa<::mlir::npu::BarrierScopeAttr>()">, "barrier scope">;
def NPU_EpilogueAttr    : Attr<CPred<"$_self.isa<::mlir::npu::EpilogueAttr>()">, "matmul epilogue policy">;
#endif
