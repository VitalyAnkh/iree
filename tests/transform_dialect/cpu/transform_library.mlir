module attributes { transform.with_named_sequence } {
  transform.named_sequence @custom_matmul(%variant_op: !transform.any_op {transform.consumed}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    %tiled_generic, %forall =
      transform.structured.tile_using_forall %0 num_threads [2]
      // TODO: IREE needs own workgroup mapping attribute.
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
        : (!transform.any_op) -> ()

    // Canonicalization/CSE is needed before bufferization otherwise unnecessary
    // allocs will be created.
    %func_op = transform.structured.match ops{["func.func"]} in %variant_op
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_op {
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    %func_op_3 = transform.iree.bufferize %func_op : (!transform.any_op) -> (!transform.any_op)
    %memref_func = transform.structured.match ops{["func.func"]} in %func_op_3
      : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()

    // CSE is needed on the workgroup_count region to pass this particular test.
    transform.apply_cse to %memref_func : !transform.any_op
    %none_attr = transform.param.constant #iree_codegen.translation_info<pipeline = None> -> !transform.any_param
    transform.annotate %memref_func "translation_info" = %none_attr : !transform.any_op, !transform.any_param
    transform.yield
  }
}
