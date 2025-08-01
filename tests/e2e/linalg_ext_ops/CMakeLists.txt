################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# tests/e2e/linalg_ext_ops/BUILD.bazel                                         #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_check_single_backend_test_suite(
  NAME
    check_llvm-cpu_local-task
  SRCS
    "arg_compare.mlir"
    "attention.mlir"
    "gather.mlir"
    "map_scatter.mlir"
    "scan.mlir"
    "scatter.mlir"
    "sort.mlir"
    "top-k.mlir"
    "winograd_input.mlir"
    "winograd_output.mlir"
  TARGET_BACKEND
    "llvm-cpu"
  DRIVER
    "local-task"
  COMPILER_FLAGS
    "--iree-llvmcpu-target-cpu=generic"
  LABELS
    "nowasm"
)

iree_check_single_backend_test_suite(
  NAME
    check_llvm-cpu_local-task_i1
  SRCS
    "attention_i1_mask.mlir"
  TARGET_BACKEND
    "llvm-cpu"
  DRIVER
    "local-task"
  COMPILER_FLAGS
    "--iree-llvmcpu-target-cpu=generic"
    "--iree-experimental-packed-i1-storage"
  LABELS
    "nowasm"
)

iree_check_single_backend_test_suite(
  NAME
    check_vmvx_local-task
  SRCS
    "arg_compare.mlir"
    "gather.mlir"
    "map_scatter.mlir"
    "scan.mlir"
    "scatter.mlir"
    "sort.mlir"
    "top-k.mlir"
    "winograd_input.mlir"
    "winograd_output.mlir"
  TARGET_BACKEND
    "vmvx"
  DRIVER
    "local-task"
)

iree_check_single_backend_test_suite(
  NAME
    check_cuda
  SRCS
    "arg_compare.mlir"
    "gather.mlir"
    "scan.mlir"
    "scatter.mlir"
    "sort.mlir"
    "top-k.mlir"
    "winograd_input.mlir"
    "winograd_output.mlir"
  TARGET_BACKEND
    "cuda"
  DRIVER
    "cuda"
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-nvidia"
)

iree_check_single_backend_test_suite(
  NAME
    check_rocm_hip
  SRCS
    "arg_compare.mlir"
    "gather.mlir"
    "map_scatter.mlir"
    "scan.mlir"
    "scatter.mlir"
    "sort.mlir"
    "winograd_input.mlir"
    "winograd_output.mlir"
  TARGET_BACKEND
    "rocm"
  DRIVER
    "hip"
)

iree_check_single_backend_test_suite(
  NAME
    check_metal-spirv_vulkan
  SRCS
    "arg_compare.mlir"
    "gather.mlir"
    "scan.mlir"
    "scatter.mlir"
    "sort.mlir"
    "winograd_input.mlir"
    "winograd_output.mlir"
  TARGET_BACKEND
    "metal-spirv"
  DRIVER
    "metal"
)

iree_check_single_backend_test_suite(
  NAME
    check_vulkan-spirv_vulkan
  SRCS
    "arg_compare.mlir"
    "gather.mlir"
    "scan.mlir"
    "scatter.mlir"
    "sort.mlir"
    "winograd_input.mlir"
    "winograd_output.mlir"
  TARGET_BACKEND
    "vulkan-spirv"
  DRIVER
    "vulkan"
)

iree_check_single_backend_test_suite(
  NAME
    check_cuda-topk-split-reduction
  SRCS
    "top-k.mlir"
  TARGET_BACKEND
    "cuda"
  DRIVER
    "cuda"
  COMPILER_FLAGS
    "--iree-dispatch-creation-topk-split-reduction=2"
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-nvidia"
)

iree_check_single_backend_test_suite(
  NAME
    check_cuda-topk-split-reduction-double
  SRCS
    "top-k.mlir"
  TARGET_BACKEND
    "cuda"
  DRIVER
    "cuda"
  COMPILER_FLAGS
    "--iree-dispatch-creation-topk-split-reduction=3,2"
  LABELS
    "noasan"
    "nomsan"
    "notsan"
    "noubsan"
    "requires-gpu-nvidia"
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
