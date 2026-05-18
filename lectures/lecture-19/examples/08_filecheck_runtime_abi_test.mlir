// RUN: mlir-opt %s --npu-rt-verify-abi --npu-rt-pack-command-buffer | FileCheck %s

// CHECK-LABEL: npu_rt.command_buffer @matmul_tile
// CHECK-SAME: abi_version = "1.0"
// CHECK: descriptor_count = 3
// CHECK: command_count = 6
// CHECK: DMA_LOAD
// CHECK: MATMUL
// CHECK: REQUANT
// CHECK: DMA_STORE
// CHECK: FENCE

// Negative tests to add with -verify-diagnostics:
// expected-error {{missing host-visible fence}}
// expected-error {{descriptor index out of range}}
// expected-error {{unsupported dtype i4_packed for target}}
