#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
module attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: memref<100xf64>, %arg1: memref<100xf64>) -> (memref<f64> {jax.result_info = ""}) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f64>
    linalg.generic {indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"]} ins(%arg0, %arg1 : memref<100xf64>, memref<100xf64>) outs(%alloc : memref<f64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %0 = arith.mulf %in, %in_0 : f64
      %1 = arith.addf %out, %0 : f64
      linalg.yield %1 : f64
    }
    return %alloc : memref<f64>
  }
}
