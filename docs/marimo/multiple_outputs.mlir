#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module @jit_multiple_outputs attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<5x2xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<2x5xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<5x5xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 2 : i32}, %arg3: tensor<2x2xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (tensor<2x2xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5x2xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}, tensor<5x5xf32> {jax.result_info = "[2]", mhlo.layout_mode = "default"}) {
    %0 = tensor.empty() : tensor<2x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg1, %arg0 : tensor<2x5xf32>, tensor<5x2xf32>) outs(%1 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in, %in_3 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<2x2xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<5x2xf32>
    %3 = tensor.empty() : tensor<5x2xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_1 : tensor<5x2xf32>, tensor<5x2xf32>) outs(%3 : tensor<5x2xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.addf %in, %in_3 : f32
      linalg.yield %8 : f32
    } -> tensor<5x2xf32>
    %5 = tensor.empty() : tensor<5x5xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %6 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<5x5xf32>) -> tensor<5x5xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<5x2xf32>, tensor<2x5xf32>) outs(%6 : tensor<5x5xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %8 = arith.mulf %in, %in_3 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<5x5xf32>

    %output_1 = bufferization.materialize_in_destination %2 in %arg3 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %output_2 = bufferization.materialize_in_destination %7 in %arg2: (tensor<5x5xf32>, tensor<5x5xf32>) -> tensor<5x5xf32>

    return %output_1, %4, %output_2 : tensor<2x2xf32>, tensor<5x2xf32>, tensor<5x5xf32>
  }
}