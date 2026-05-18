// Pseudo C++ skeleton for an out-of-tree NPU Transform Dialect extension.
// This is a teaching artifact, not a drop-in build file.

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace npu {
class NPUTransformDialectExtension
  : public mlir::transform::TransformDialectExtension<NPUTransformDialectExtension> {
public:
  NPUTransformDialectExtension() {
    // declareGeneratedDialect<npu::NPUKernelDialect>();
    // registerTransformOps<ScheduleMatmulTileOp, PromoteToSramOp, TensorizeToNpuOp>();
  }
};
}

// Design sketch:
// npu.schedule.matmul_tile %target {m=16, n=16, k=32}
// npu.schedule.promote_to_sram %tile {space="SRAM0", double_buffer=true}
// npu.schedule.tensorize_to_npu %vector_contract {instruction="mma_i8_i32"}
