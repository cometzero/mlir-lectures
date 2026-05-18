// Lecture 11 - TypeConverter skeleton for future tensor/memref boundary.
// In many NPU compilers, graph legalization keeps tensor types and kernel
// legalization/bufferization introduces memref + memory space. This skeleton
// shows where the policy can live.

class NPUTypeConverter : public TypeConverter {
public:
  NPUTypeConverter(MLIRContext *ctx) {
    // Identity conversion for types that are already legal.
    addConversion([](Type type) { return type; });

    // Example: convert ranked tensor to a packed memref descriptor later.
    addConversion([ctx](RankedTensorType tensorTy) -> std::optional<Type> {
      if (!tensorTy.hasStaticShape()) return std::nullopt;
      auto elemTy = tensorTy.getElementType();
      auto layout = npu::getPackedLayoutAttr(ctx, tensorTy.getShape());
      unsigned memorySpace = 1; // 1 = NPU SRAM in this pseudo ABI.
      return MemRefType::get(tensorTy.getShape(), elemTy, layout, memorySpace);
    });

    // Source/target materializations are needed when converted and unconverted
    // regions temporarily meet. Prefer minimizing these bridges.
  }
};
