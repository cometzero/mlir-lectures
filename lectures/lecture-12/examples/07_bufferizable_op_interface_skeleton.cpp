// Educational skeleton only. Adjust API names to your LLVM/MLIR revision.

struct NpuMatmulBufferizableOpInterface
    : public BufferizableOpInterface::ExternalModel<
          NpuMatmulBufferizableOpInterface, npu::MatmulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // A and B are read; destination/accumulator may be read for accumulation.
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // The destination/accumulator operand is written.
    return isDestinationOperand(op, opOperand);
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    // Result aliases with destination if legal.
    if (isDestinationOperand(op, opOperand))
      return {{op->getResult(0), BufferRelation::Equivalent}};
    return {};
  }
};
