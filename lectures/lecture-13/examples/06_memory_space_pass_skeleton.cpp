// Lecture 13 - Pseudo pass skeleton. Not drop-in compilable.
struct AssignNpuMemorySpacesPass
    : public PassWrapper<AssignNpuMemorySpacesPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "npu-assign-memory-spaces"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  void runOnOperation() override {
    // 1. Classify buffers: DRAM/SRAM/ACCUM/CONST.
    // 2. Analyze lifetime and capacity.
    // 3. Assign layout and bank attributes.
    // 4. Diagnose unsupported layout or illegal escape.
  }
};
