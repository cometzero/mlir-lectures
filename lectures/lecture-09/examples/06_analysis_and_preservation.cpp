// Pseudo analysis and preservation pattern.
struct NpuSramFootprintAnalysis {
  explicit NpuSramFootprintAnalysis(Operation *op) {
    // Walk candidate kernels and compute tile buffer footprint.
  }
  bool fits(Operation *candidate) const { return true; }
};

struct NpuTilePass : PassWrapper<NpuTilePass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto &footprint = getAnalysis<NpuSramFootprintAnalysis>();
    func::FuncOp f = getOperation();

    bool changed = false;
    f.walk([&](Operation *op) {
      if (/* candidate */ false) {
        if (!footprint.fits(op))
          return signalPassFailure();
        // Rewrite to tiled loop/kernel form.
        changed = true;
      }
    });

    if (!changed) {
      markAllAnalysesPreserved();
      return;
    }
    // Do not preserve footprint analysis after tiling: the IR structure changed.
    markAnalysesPreserved<DominanceInfo>(); // only if dominance remains valid.
  }
};
