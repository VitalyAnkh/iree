################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Dialect/VM/Transforms/test/BUILD.bazel            #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "deduplicate_rodata.mlir"
    "drop_empty_module_initializers.mlir"
    "drop_unused_calls.mlir"
    "global_initialization.mlir"
    "hoist_inlined_rodata.mlir"
    "ordinal_allocation.mlir"
    "reify_rodata_tables.mlir"
    "resolve_rodata_loads.mlir"
    "sink_defining_ops.mlir"
  TOOLS
    FileCheck
    iree-opt
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
