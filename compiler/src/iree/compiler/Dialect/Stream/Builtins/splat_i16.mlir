// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// stream.builtin.splat.i16
// Writes the i16 %value %count times at offset 0 of %out_binding.

stream.executable private @__builtin_splat_i16 {
  stream.executable.export public @__builtin_splat_i16 workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @__builtin_splat_i16(%value: i16, %count: index, %out_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %count0 = iree_tensor_ext.dispatch.workload.ordinal %count, 0 : index
      %out = stream.binding.subspan %out_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi16>>{%count0}
      %0 = tensor.empty(%count0) : tensor<?xi16>
      %1 = linalg.fill ins(%value : i16) outs(%0 : tensor<?xi16>) -> tensor<?xi16>
      iree_tensor_ext.dispatch.tensor.store %1, %out, offsets = [0], sizes = [%count0], strides = [1] : tensor<?xi16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi16>>{%count}
      return
    }
  }
}
