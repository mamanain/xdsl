from xdsl.dialects.builtin import (
    AnyAttr,
    AnyTensorTypeConstr,
    IntAttr,
    IntegerAttr,
    IntegerType,
)
from xdsl.irdl import (
    AllOf,
    AnyOf,
    BaseAttr,
    EqAttrConstraint,
    OpDef,
    OperandDef,
    ParamAttrConstraint,
    ParamAttrDef,
    PropertyDef,
    ResultDef,
)
from xdsl.utils.dialect_dump import dump_dialect_pyfile, generate_dynamic_attr_class

types = [
    ("Test_SingletonAType", ParamAttrDef(name="test.singleton_a", parameters=[])),
    ("Test_SingletonBType", ParamAttrDef(name="test.singleton_b", parameters=[])),
    ("Test_SingletonCType", ParamAttrDef(name="test.singleton_c", parameters=[])),
]

attrs = [("Test_TestAttr", ParamAttrDef(name="test.test", parameters=[]))]

BaseAttr(generate_dynamic_attr_class(types[0][0], types[0][1]))

ops = [
    (
        "Test_AndOp",
        OpDef(
            name="test.and",
            operands=[
                (
                    "in_",
                    OperandDef(
                        AllOf(
                            (
                                AnyAttr(),
                                BaseAttr(
                                    generate_dynamic_attr_class(
                                        types[0][0], types[0][1]
                                    )
                                ),
                            )
                        )
                    ),
                )
            ],
        ),
    ),
    ("Test_AnyOp", OpDef(name="test.any", operands=[("in_", OperandDef(AnyAttr()))])),
    (
        "Test_AttributesOp",
        OpDef(
            name="test.attributes",
            properties={
                "int_attr": PropertyDef(
                    IntegerAttr.constr(type=EqAttrConstraint(IntegerType(16)))
                ),
                "in": PropertyDef(AnyAttr()),
            },
            accessor_names={"in_": ("in", "property")},
        ),
    ),
    (
        "Test_Integers",
        OpDef(
            name="test.integers",
            operands=[
                (
                    "any_int",
                    OperandDef(
                        ParamAttrConstraint(
                            IntegerType,
                            (EqAttrConstraint(IntAttr(8)), AnyAttr()),
                        )
                    ),
                ),
                ("any_integer", OperandDef(BaseAttr(IntegerType))),
            ],
        ),
    ),
    (
        "Test_OrOp",
        OpDef(
            name="test.or",
            operands=[
                (
                    "in_",
                    OperandDef(
                        AnyOf(
                            (
                                BaseAttr(
                                    generate_dynamic_attr_class(
                                        types[0][0], types[0][1]
                                    )
                                ),
                                BaseAttr(
                                    generate_dynamic_attr_class(
                                        types[1][0], types[1][1]
                                    )
                                ),
                                BaseAttr(
                                    generate_dynamic_attr_class(
                                        types[2][0], types[2][1]
                                    )
                                ),
                            )
                        )
                    ),
                )
            ],
        ),
    ),
    (
        "Test_TypesOp",
        OpDef(name="test.types", operands=[("in_", OperandDef(AnyAttr()))]),
    ),
    (
        "Test_SingleOp",
        OpDef(
            name="test.single",
            operands=[("arg", OperandDef(AnyTensorTypeConstr))],
            results=[("res", ResultDef(AnyTensorTypeConstr))],
            assembly_format="$arg attr-dict : type($arg) -> type($res)",
        ),
    ),
]

dump_dialect_pyfile(
    "test",
    ops,
    attrs,
    types,
    "tests/dialect_dump/generated_dialect.py",
    "tests/dialect_dump/test_dialect_dump.py",
)
