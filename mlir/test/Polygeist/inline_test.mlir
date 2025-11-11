// RUN: mlir-opt --pocl-inliner --allow-unregistered-dialect --split-input-file %s | FileCheck %s

module {
  func.func private @_Z13get_global_idj(%arg0: i32) -> i64 {
    %0 = call @_Z12get_group_idj(%arg0) : (i32) -> i64
    %1 = call @_Z14get_local_sizej(%arg0) : (i32) -> i64
    %2 = arith.muli %0, %1 : i64
    %3 = call @_Z12get_local_idj(%arg0) : (i32) -> i64
    %4 = arith.addi %2, %3 : i64
    return %4 : i64
  }
  func.func private @_Z12get_group_idj(%arg0: i32) -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.index_switch %0 -> i64
    case 0 {
      %block_id_x = gpu.block_id  x
      %2 = arith.index_cast %block_id_x : index to i64
      scf.yield %2 : i64
    }
    case 1 {
      %block_id_y = gpu.block_id  y
      %2 = arith.index_cast %block_id_y : index to i64
      scf.yield %2 : i64
    }
    case 2 {
      %block_id_z = gpu.block_id  z
      %2 = arith.index_cast %block_id_z : index to i64
      scf.yield %2 : i64
    }
    default {
      scf.yield %c0_i64 : i64
    }
    return %1 : i64
  }
  func.func private @_Z14get_local_sizej(%arg0: i32) -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.index_switch %0 -> i64
    case 0 {
      %block_dim_x = gpu.block_dim  x
      %2 = arith.index_cast %block_dim_x : index to i64
      scf.yield %2 : i64
    }
    case 1 {
      %block_dim_y = gpu.block_dim  y
      %2 = arith.index_cast %block_dim_y : index to i64
      scf.yield %2 : i64
    }
    case 2 {
      %block_dim_z = gpu.block_dim  z
      %2 = arith.index_cast %block_dim_z : index to i64
      scf.yield %2 : i64
    }
    default {
      scf.yield %c0_i64 : i64
    }
    return %1 : i64
  }
  func.func private @_Z12get_local_idj(%arg0: i32) -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.index_switch %0 -> i64
    case 0 {
      %thread_id_x = gpu.thread_id  x
      %2 = arith.index_cast %thread_id_x : index to i64
      scf.yield %2 : i64
    }
    case 1 {
      %thread_id_y = gpu.thread_id  y
      %2 = arith.index_cast %thread_id_y : index to i64
      scf.yield %2 : i64
    }
    case 2 {
      %thread_id_z = gpu.thread_id  z
      %2 = arith.index_cast %thread_id_z : index to i64
      scf.yield %2 : i64
    }
    default {
      scf.yield %c0_i64 : i64
    }
    return %1 : i64
  }
  func.func @vecadd_kernel(%arg0: memref<?xi32>, %arg1: i32) attributes {gpu.kernel} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = call @_Z13get_global_idj(%c0_i32) : (i32) -> i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = memref.load %arg0[%2] : memref<?xi32>
    %4 = arith.addi %3, %c1_i32 : i32
    memref.store %4, %arg0[%2] : memref<?xi32>
    return
  }
}

// 			CHECK:	func.func @vecadd_kernel(%[[arg0:.+]]: memref<?xi32>, %[[arg1:.+]]: i32) attributes {gpu.kernel} {
// CHECK-DAG:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:    %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK:    %[[x0:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:      %[[x5:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:        %[[x6:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x11:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x12:.+]] = arith.index_cast %[[c0_i32]] : i32 to index
// CHECK-NEXT:            %[[x13:.+]] = scf.index_switch %[[x12]] -> i64
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[block_id_x:.+]] = gpu.block_id  x
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[block_id_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[block_id_y:.+]] = gpu.block_id  y
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[block_id_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[block_id_z:.+]] = gpu.block_id  z
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[block_id_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64:.+]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x13]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x11]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x7:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x11:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x12:.+]] = arith.index_cast %[[c0_i32]] : i32 to index
// CHECK-NEXT:            %[[x13:.+]] = scf.index_switch %[[x12]] -> i64
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[block_dim_x:.+]] = gpu.block_dim  x
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[block_dim_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[block_dim_y:.+]] = gpu.block_dim  y
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[block_dim_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[block_dim_z:.+]] = gpu.block_dim  z
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[block_dim_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x13]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x11]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x8:.+]] = arith.muli %[[x6]], %[[x7]] : i64
// CHECK-NEXT:        %[[x9:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x11:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x12:.+]] = arith.index_cast %[[c0_i32]] : i32 to index
// CHECK-NEXT:            %[[x13:.+]] = scf.index_switch %[[x12]] -> i64
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[thread_id_x:.+]] = gpu.thread_id  x
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[thread_id_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[thread_id_y:.+]] = gpu.thread_id  y
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[thread_id_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[thread_id_z:.+]] = gpu.thread_id  z
// CHECK-NEXT:              %[[x14:.+]] = arith.index_cast %[[thread_id_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x14]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x13]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x11]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x10:.+]] = arith.addi %[[x8]], %[[x9]] : i64
// CHECK-NEXT:        scf.yield %[[x10]] : i64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.alloca_scope.return %[[x5]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[x1:.+]] = arith.trunci %[[x0]] : i64 to i32
// CHECK-NEXT:    %[[x2:.+]] = arith.index_cast %[[x1]] : i32 to index
// CHECK-NEXT:    %[[x3:.+]] = memref.load %[[arg0]][%[[x2]]] : memref<?xi32>
// CHECK-NEXT:    %[[x4:.+]] = arith.addi %[[x3]], %[[c1_i32]] : i32
// CHECK-NEXT:    memref.store %[[x4]], %[[arg0]][%[[x2]]] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
