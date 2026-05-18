// Pseudo pipeline registration.
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

struct NpuLoweringPipelineOptions
    : public PassPipelineOptions<NpuLoweringPipelineOptions> {
  Option<int> sramKb{*this, "sram-kb", llvm::cl::desc("SRAM size in KiB"), llvm::cl::init(512)};
  Option<int> tileM{*this, "tile-m", llvm::cl::desc("M tile"), llvm::cl::init(64)};
  Option<int> tileN{*this, "tile-n", llvm::cl::desc("N tile"), llvm::cl::init(128)};
  Option<int> tileK{*this, "tile-k", llvm::cl::desc("K tile"), llvm::cl::init(32)};
};

void buildNpuLoweringPipeline(OpPassManager &pm,
                              const NpuLoweringPipelineOptions &opts) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  OpPassManager &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(createNpuDecomposeCompositesPass());
  funcPM.addPass(createFuseMatmulEpiloguePass());

  pm.addPass(createNpuInferLayoutPass());
  pm.addPass(createNpuTileLinalgPass(opts.tileM, opts.tileN, opts.tileK));
  pm.addPass(createOneShotBufferizePass());
  pm.addPass(createNpuAssignMemorySpacePass(opts.sramKb));
  pm.addPass(createNpuInsertDmaPass(/*doubleBuffer=*/true));
  pm.addPass(createNpuPlaceBarriersPass());
  pm.addPass(createNpuEmitRuntimeAbiPass());
}

void registerNpuPipelines() {
  PassPipelineRegistration<NpuLoweringPipelineOptions>(
      "npu-lowering-pipeline",
      "Lower tensor/linalg IR to NPU runtime ABI",
      buildNpuLoweringPipeline);
}
