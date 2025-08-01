// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldAllocatorSelect1
// CHECK-SAME: (%[[SAME_DEVICE:.+]]: !hal.device, %[[SAME_AFFINITY:.+]]: i64)
util.func public @FoldAllocatorSelect1(%same_device: !hal.device, %same_affinity: i64) -> (!hal.device, i64) {
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  // CHECK-NOT: hal.allocator.select
  %device, %queue_affinity = hal.allocator.select
      from([
        (%same_device, %same_affinity : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[SAME_DEVICE]], %[[SAME_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectSameDevice
// CHECK-SAME: (%[[SAME_DEVICE:.+]]: !hal.device, %[[AFFINITY_A:.+]]: i64, %[[AFFINITY_B:.+]]: i64)
util.func public @FoldAllocatorSelectSameDevice(%same_device: !hal.device, %affinity_a: i64, %affinity_b: i64) -> (!hal.device, i64) {
  // CHECK: %[[UNUSED:.+]], %[[QUEUE_AFFINITY:.+]] = hal.allocator.select
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  %device, %queue_affinity = hal.allocator.select
      from([
        (%same_device, %affinity_a : !hal.device, i64),
        (%same_device, %affinity_b : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[SAME_DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectSameQueueAffinity
// CHECK-SAME: (%[[DEVICE_A:.+]]: !hal.device, %[[DEVICE_B:.+]]: !hal.device, %[[SAME_AFFINITY:.+]]: i64)
util.func public @FoldAllocatorSelectSameQueueAffinity(%device_a: !hal.device, %device_b: !hal.device, %same_affinity: i64) -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]], %[[UNUSED:.+]] = hal.allocator.select
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  %device, %queue_affinity = hal.allocator.select
      from([
        (%device_a, %same_affinity : !hal.device, i64),
        (%device_b, %same_affinity : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[SAME_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}
