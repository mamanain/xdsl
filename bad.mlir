#map = affine_map<() -> ()>
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: memref<100xf64>, %arg1: memref<100xf64>) -> (memref<f64> {jax.result_info = ""}) {
    %cst = arith.constant 0.000000e+00 : f64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = []} ins(%cst : f64) outs(%alloc : memref<f64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    }
    return %alloc : memref<f64>
  }
}
