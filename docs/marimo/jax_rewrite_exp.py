import marimo

__generated_with = "0.8.5"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    original_func = """
    #map = affine_map<(d0, d1, d2) -> (d0, d2)>
    #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
    #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

    func.func main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<3x4xf32> {mhlo.layout_mode = "default"}, %arg2: tensor<2x4xf32> {mhlo.layout_mode = "default", tf.aliasing_output = 0 : i32}) -> (tensor<2x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
        %0 = tensor.empty() : tensor<2x4xf32>
        %cst = arith.constant 0.000000e+00 : f32
        %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
        %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>) outs(%1 : tensor<2x4xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %3 = arith.mulf %in, %in_0 : f32
          %4 = arith.addf %out, %3 : f32
          linalg.yield %4 : f32
        } -> tensor<2x4xf32>
        return %2 : tensor<2x4xf32>
      }
    """
    return original_func,


@app.cell
def __():
    from xdsl.context import MLContext
    from xdsl.parser import Parser
    return MLContext, Parser


@app.cell
def __(MLContext, Parser, original_func):
    ctx = MLContext()
    parser = Parser(ctx, original_func)
    return ctx, parser


@app.cell
def __(parser):
    parser.parse_module(True)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()