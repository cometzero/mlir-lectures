//===- NPUOps.cpp - verifier skeleton ------------------------------------===//
#include "NPU/IR/NPUOps.h"
#include "mlir/IR/Diagnostics.h"
using namespace mlir;
using namespace mlir::npu;

static bool isSupportedTileShape(int64_t m, int64_t n, int64_t k) {
  return m == 16 && n == 16 && (k == 32 || k == 64);
}

LogicalResult MatmulOp::verify() {
  auto tile = getTile();
  if (!isSupportedTileShape(tile.getM(), tile.getN(), tile.getK()))
    return emitOpError("unsupported native tile shape ")
      << "m=" << tile.getM() << ", n=" << tile.getN() << ", k=" << tile.getK();
  if (getLhsLayout().getMnemonic() != "mk")
    return emitOpError("lhs_layout must be #npu.layout<mk>");
  if (getRhsLayout().getMnemonic() != "kn")
    return emitOpError("rhs_layout must be #npu.layout<kn>");
  // TODO: check element type pair, accumulator type, memory spaces, epilogue.
  return success();
}

LogicalResult DmaOp::verify() {
  if (getElements() <= 0)
    return emitOpError("elements must be positive");
  // TODO: verify direction, alignment, max transfer size, stride policy.
  return success();
}

LogicalResult BarrierOp::verify() {
  if (getTokens().empty())
    return emitOpError("requires at least one token operand");
  // TODO: verify scope compatibility and command-region dependencies.
  return success();
}
