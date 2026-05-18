// Custom parser/printer skeleton. Prefer declarative assemblyFormat when possible.
ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &state) {
  // 1. parse lhs, rhs, acc operands
  // 2. parse attr-dict with tile/layout/epilogue
  // 3. parse operand/result types
  // 4. resolve operands and add result type
  return parser.emitError(parser.getCurrentLocation(), "custom parser skeleton");
}

void MatmulOp::print(OpAsmPrinter &printer) {
  printer << " " << getLhs() << ", " << getRhs() << ", " << getAcc();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getLhs().getType() << ", " << getRhs().getType()
          << ", " << getAcc().getType() << " -> " << getResult().getType();
}
