# RUN: python %s | filecheck %s

from xdsl.dialects.arith import AddfOp, MulfOp, SubfOp
from xdsl.dialects.bigint import AddOp, BigIntegerType, MulOp, SubOp
from xdsl.dialects.builtin import Float64Type
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.program import FrontendProgram

p1 = FrontendProgram()
p1.register_type(int, BigIntegerType)
p1.register_function(int.__add__, AddOp)
p1.register_function(int.__sub__, SubOp)
p1.register_function(int.__mul__, MulOp)
with CodeContext(p1):
    # CHECK:      builtin.module {
    # CHECK-NEXT:     func.func @foo(%x : !bigint.bigint, %y : !bigint.bigint, %z : !bigint.bigint) -> !bigint.bigint {
    # CHECK-NEXT:       %0 = bigint.mul %y, %z : !bigint.bigint
    # CHECK-NEXT:       %1 = bigint.sub %x, %0 : !bigint.bigint
    # CHECK-NEXT:       func.return %1 : !bigint.bigint
    # CHECK-NEXT:     }
    # CHECK-NEXT:   }
    def foo(x: int, y: int, z: int) -> int:
        return x - y * z


p1.compile()
print(p1.textual_format())


p2 = FrontendProgram()
p2.register_type(float, Float64Type)
p2.register_function(float.__add__, AddfOp)
p2.register_function(float.__sub__, SubfOp)
p2.register_function(float.__mul__, MulfOp)
with CodeContext(p2):
    # CHECK:      builtin.module {
    # CHECK-NEXT:       func.func @bar(%x : f64, %y : f64, %z : f64) -> f64 {
    # CHECK-NEXT:         %0 = arith.mulf %y, %z : f64
    # CHECK-NEXT:         %1 = arith.subf %x, %0 : f64
    # CHECK-NEXT:         func.return %1 : f64
    # CHECK-NEXT:       }
    # CHECK-NEXT:     }
    def bar(x: float, y: float, z: float) -> float:
        return x - y * z


p2.compile()
print(p2.textual_format())
