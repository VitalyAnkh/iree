// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-softmax),cse)" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-softmax{use-fusion=true}),cse)" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-softmax{use-fusion=false}),cse)" %s | FileCheck %s --check-prefix=CHECK-NO-FUSE

func.func @softmax(%arg0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
  %0 = tensor.empty() : tensor<2x16x32xf32>
  %1 = linalg.softmax dimension(2) ins(%arg0 : tensor<2x16x32xf32>) outs(%0: tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
  return %1 : tensor<2x16x32xf32>
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-DAG:    %[[D0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK-DAG:    %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-DAG:    %[[CST:.+]] = arith.constant 0xFFC00000 : f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 0.0
// CHECK:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK:          linalg.yield %[[D8]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D4:.+]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK:        %[[D5:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[D3]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[D4]] : tensor<2x16xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK:          %[[D10:.+]] = arith.addf %[[D9]], %[[OUT]]
// CHECK:          linalg.yield %[[D10]] : f32
// CHECK:        } -> tensor<2x16xf32>
// CHECK:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]], #[[MAP]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[ARG0]], %[[D3]], %[[D5]] : tensor<2x16x32xf32>, tensor<2x16xf32>, tensor<2x16xf32>)
// CHECK-SAME:     outs(%[[D0]] : tensor<2x16x32xf32>) {
// CHECK:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[IN_2:.+]]: f32, %[[OUT0:.+]]: f32):
// CHECK:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK:          %[[D10:.+]] = arith.divf %[[D9]], %[[IN_2]] : f32
// CHECK:          linalg.yield %[[D10]] : f32
// CHECK:        } -> tensor<2x16x32xf32>
// CHECK:        return %[[D7]] : tensor<2x16x32xf32>
// CHECK:      }

// CHECK-NO-FUSE-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NO-FUSE-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-NO-FUSE:      func.func @softmax(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK-NO-FUSE-DAG:     %[[D0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK-NO-FUSE-DAG:     %[[D1:.+]] = tensor.empty() : tensor<2x16xf32>
// CHECK-NO-FUSE-DAG:     %[[CST:.+]] = arith.constant 0xFFC00000 : f32
// CHECK-NO-FUSE-DAG:     %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NO-FUSE:        %[[D2:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D3:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types = ["parallel",
// CHECK-NO-FUSE-SAME:     "parallel", "reduction"]} ins(%[[ARG0]] : tensor<2x16x32xf32>) outs(%[[D2]] : tensor<2x16xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK-NO-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-NO-FUSE:        } -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D4:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP]]], iterator_types = ["parallel",
// CHECK-NO-FUSE-SAME:     "parallel", "parallel"]} ins(%[[ARG0]], %3 : tensor<2x16x32xf32>, tensor<2x16xf32>) outs(%[[D0]] : tensor<2x16x32xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.subf %[[IN]], %[[IN_1]] : f32
// CHECK-NO-FUSE:          %[[D9:.+]] = math.exp %[[D8]] : f32
// CHECK-NO-FUSE:          linalg.yield %[[D9]] : f32
// CHECK-NO-FUSE:        } -> tensor<2x16x32xf32>
// CHECK-NO-FUSE:        %[[D5:.+]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[D1]] : tensor<2x16xf32>) -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D6:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]], iterator_types =
// CHECK-NO-FUSE-SAME:     ["parallel", "parallel", "reduction"]} ins(%[[D4]] : tensor<2x16x32xf32>)
// CHECK-NO-FUSE-SAME:     outs(%[[D5]] : tensor<2x16xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.addf %[[IN]], %[[OUT]]
// CHECK-NO-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-NO-FUSE:        } -> tensor<2x16xf32>
// CHECK-NO-FUSE:        %[[D7:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP]]], iterator_types =
// CHECK-NO-FUSE-SAME:     ["parallel", "parallel", "parallel"]} ins(%[[D4]], %[[D6]] : tensor<2x16x32xf32>, tensor<2x16xf32>)
// CHECK-NO-FUSE-SAME:     outs(%[[D0]] : tensor<2x16x32xf32>) {
// CHECK-NO-FUSE:        ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NO-FUSE:          %[[D8:.+]] = arith.divf %[[IN]], %[[IN_1]] : f32
// CHECK-NO-FUSE:          linalg.yield %[[D8]] : f32
// CHECK-NO-FUSE:        } -> tensor<2x16x32xf32>
// CHECK-NO-FUSE:        return %[[D7]] : tensor<2x16x32xf32>
// CHECK-NO-FUSE:      }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @do_not_fuse_gather(%arg0: tensor<4096x64xi64>, %arg1: tensor<4096x64xf32>) -> tensor<4096x64xf32> {
  %empty = tensor.empty() : tensor<4096x64xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4096x64xi64>) outs(%empty : tensor<4096x64xf32>) {
  ^bb0(%in: i64, %out: f32):
    %3 = linalg.index 0 : index
    %4 = arith.index_cast %in : i64 to index
    %extracted = tensor.extract %arg1[%3, %4] : tensor<4096x64xf32>
    linalg.yield %extracted : f32
  } -> tensor<4096x64xf32>
  %s_empty = tensor.empty() : tensor<4096x64xf32>
  %1 = linalg.softmax dimension(1) ins(%0 : tensor<4096x64xf32>) outs(%s_empty: tensor<4096x64xf32>) -> tensor<4096x64xf32>
  return %1 : tensor<4096x64xf32>
}
//   CHECK-LABEL: func @do_not_fuse_gather(
//         CHECK:    linalg.generic {{.*}}
//         CHECK:      tensor.extract {{.*}} : tensor<4096x64xf32>
// CHECK-COUNT-3:    linalg.generic
