// Lecture 10 - PDL-style sketch for discussing declarative matching.
// This is a teaching sketch; use actual ODS/PDL constraints for production.

module {
  pdl.pattern @fuse_matmul_bias_relu : benefit(10) {
    %relu_name = pdl.attribute = "linalg.generic"
    %add_name = pdl.attribute = "linalg.generic"
    %mm_name = pdl.attribute = "linalg.generic"

    %relu = pdl.operation %relu_name
    %add_result = pdl.operand of %relu
    %add = pdl.operation %add_name -> (%add_result)
    %mm_result = pdl.operand of %add
    %mm = pdl.operation %mm_name -> (%mm_result)

    pdl.apply_native_constraint "isReluGeneric"(%relu)
    pdl.apply_native_constraint "isAddBiasGeneric"(%add)
    pdl.apply_native_constraint "isMatMulGeneric"(%mm)
    pdl.apply_native_constraint "isNPULegalEpilogue"(%mm, %add, %relu)

    pdl.rewrite %relu with "rewriteToNPUMatmulEpilogue"(%mm, %add, %relu)
  }
}
