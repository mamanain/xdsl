from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, cast

from typing_extensions import TypeVar

from xdsl.dialect_interfaces import OpAsmDialectInterface
from xdsl.dialects.builtin import (
    AffineMapAttr,
    AffineSetAttr,
    AnyFloat,
    AnyUnrankedMemRefType,
    AnyUnrankedTensorType,
    AnyVectorType,
    ArrayAttr,
    BFloat16Type,
    BoolAttr,
    BuiltinAttribute,
    BytesAttr,
    ComplexElementCovT,
    ComplexType,
    DenseArrayBase,
    DenseResourceAttr,
    DictionaryAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    FunctionType,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    LocationAttr,
    MemRefType,
    NoneAttr,
    NoneType,
    OpaqueAttr,
    Signedness,
    StridedLayoutAttr,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    UnitAttr,
    UnrankedMemRefType,
    UnrankedTensorType,
    UnregisteredAttr,
    UnregisteredOp,
    VectorType,
    i1,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Data,
    Dialect,
    OpaqueSyntaxAttribute,
    Operation,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
    TypedAttribute,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator
from xdsl.utils.base_printer import BasePrinter
from xdsl.utils.bitwise_casts import (
    convert_f16_to_u16,
    convert_f32_to_u32,
    convert_f64_to_u64,
)
from xdsl.utils.diagnostic import Diagnostic
from xdsl.utils.hints import isa
from xdsl.utils.mlir_lexer import MLIRLexer


@dataclass(eq=False, repr=False)
class Printer(BasePrinter):
    print_generic_format: bool = field(default=False)
    print_properties_as_attributes: bool = field(default=False)
    print_debuginfo: bool = field(default=False)
    diagnostic: Diagnostic = field(default_factory=Diagnostic)

    _ssa_values: dict[SSAValue, str] = field(
        default_factory=dict[SSAValue, str], init=False
    )
    """
    maps SSA Values to their "allocated" names
    """
    _ssa_names: list[dict[str, int]] = field(
        default_factory=lambda: [dict[str, int]()], init=False
    )
    _block_names: list[dict[Block, int]] = field(
        default_factory=lambda: [dict[Block, int]()], init=False
    )
    _next_valid_name_id: list[int] = field(default_factory=lambda: [0], init=False)
    _next_valid_block_id: list[int] = field(default_factory=lambda: [0], init=False)

    _dialect_resources: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    """
    resources that were referenced in the ir
    """

    @property
    def ssa_names(self):
        return self._ssa_names[-1]

    @property
    def block_names(self):
        return self._block_names[-1]

    @contextmanager
    def in_angle_brackets(self):
        self.print_string("<")
        try:
            yield
        finally:
            self.print_string(">")

    def print(self, *argv: Any) -> None:
        for arg in argv:
            if isinstance(arg, str):
                self.print_string(arg)
                continue
            if isinstance(arg, SSAValue):
                arg = cast(SSAValue[Attribute], arg)
                self.print_ssa_value(arg)
                continue
            if isinstance(arg, Attribute):
                self.print_attribute(arg)
                continue
            if isinstance(arg, Region):
                self.print_region(arg)
                continue
            if isinstance(arg, Block):
                self.print_block(arg)
                self._print_new_line()
                continue
            if isinstance(arg, Operation):
                self.print_op(arg)
                self._print_new_line()
                continue

            text = str(arg)
            self.print_string(text)

    K = TypeVar("K")
    V = TypeVar("V")

    def print_dictionary(
        self,
        elems: dict[K, V],
        print_key: Callable[[K], None],
        print_value: Callable[[V], None],
        delimiter: str = ", ",
    ) -> None:
        for i, (key, value) in enumerate(elems.items()):
            if i:
                self.print_string(delimiter)
            print_key(key)
            self.print_string("=")
            print_value(value)

    def _get_new_valid_name_id(self) -> str:
        self._next_valid_name_id[-1] += 1
        return str(self._next_valid_name_id[-1] - 1)

    def _get_new_valid_block_id(self) -> int:
        self._next_valid_block_id[-1] += 1
        return self._next_valid_block_id[-1] - 1

    def _print_results(self, op: Operation) -> None:
        results = op.results
        # No results
        if len(results) == 0:
            return

        # Multiple results
        self.print_list(op.results, self.print)
        self.print_string(" = ")

    def print_ssa_value(self, value: SSAValue) -> str:
        """
        Print an SSA value in the printer. This assigns a name to the value if the value
        does not have one in the current printing context.
        If the value has a name hint, it will use it as a prefix, and otherwise assign
        a number as the name. Numbers are assigned in order.

        Returns the name used for printing the value.
        """
        if value in self._ssa_values:
            name = self._ssa_values[value]
        elif value.name_hint:
            curr_ind = self.ssa_names.get(value.name_hint, 0)
            suffix = f"_{curr_ind}" if curr_ind != 0 else ""
            name = f"{value.name_hint}{suffix}"
            self._ssa_values[value] = name
            self.ssa_names[value.name_hint] = curr_ind + 1
        else:
            name = self._get_new_valid_name_id()
            self._ssa_values[value] = name

        self.print_string(f"%{name}")
        return name

    def print_operand(self, operand: SSAValue) -> None:
        self.print_ssa_value(operand)

    def print_block_name(self, block: Block) -> None:
        self.print_string("^")
        if block not in self.block_names:
            self.block_names[block] = self._get_new_valid_block_id()
        self.print_string(f"{self._block_names[-1][block]}")

    def print_block(
        self,
        block: Block,
        print_block_args: bool = True,
        print_block_terminator: bool = True,
    ) -> None:
        """
        Print a block with syntax `(<caret-ident>`(` <block-args> `)`)? ops* )`
        * If `print_block_args` is False, the label and arguments are not printed.
        * If `print_block_terminator` is False, the block terminator is not printed.
        """

        if print_block_args:
            self._print_new_line()
            self.print_block_name(block)
            if len(block.args) != 0:
                self.print_string("(")
                self.print_list(block.args, self.print_block_argument)
                self.print_string(")")
            self.print_string(":")

        with self.indented():
            for op in block.ops:
                if not print_block_terminator and op.has_trait(
                    IsTerminator, value_if_unregistered=False
                ):
                    continue
                self._print_new_line()
                self.print_op(op)

    def print_block_argument(self, arg: BlockArgument, print_type: bool = True) -> None:
        """
        Print a block argument with its type, e.g. `%arg : i32`
        Optionally, do not print the type.
        """
        self.print_ssa_value(arg)
        if print_type:
            self.print_string(" : ")
            self.print_attribute(arg.type)
            if self.print_debuginfo:
                self.print_string(" loc(unknown)")

    def print_region(
        self,
        region: Region,
        print_entry_block_args: bool = True,
        print_empty_block: bool = True,
        print_block_terminators: bool = True,
    ) -> None:
        """
        Print a region with syntax `{ <block>* }`
        * If `print_entry_block_args` is False, the arguments of the entry block
          are not printed.
        * If `print_empty_block` is False, empty entry blocks are not printed.
        * If `print_block_terminators` is False, the block terminators are not printed.
        """
        # Empty region
        self.print_string("{")
        if (entry_block := region.blocks.first) is None:
            self._print_new_line()
            self.print_string("}")
            return

        print_entry_block_args = (
            bool(entry_block.args) and print_entry_block_args
        ) or (not entry_block.ops and print_empty_block)
        self.print_block(
            entry_block,
            print_block_args=print_entry_block_args,
            print_block_terminator=print_block_terminators,
        )

        next_block = entry_block.next_block
        while next_block is not None:
            self.print_block(next_block, print_block_terminator=print_block_terminators)
            next_block = next_block.next_block

        self._print_new_line()
        self.print_string("}")

    def print_regions(self, regions: Sequence[Region]) -> None:
        if len(regions) == 0:
            return

        self.print_string(" (")
        self.print_list(regions, self.print_region)
        self.print_string(")")

    def print_operands(self, operands: Sequence[SSAValue]) -> None:
        self.print_string("(")
        self.print_list(operands, self.print_operand)
        self.print_string(")")

    def print_paramattr_parameters(
        self, params: Sequence[Attribute], always_print_brackets: bool = False
    ) -> None:
        if len(params) == 0 and not always_print_brackets:
            return
        self.print_string("<")
        self.print_list(params, self.print_attribute)
        self.print_string(">")

    def print_string_literal(self, string: str):
        self.print_string(json.dumps(string))

    def print_identifier_or_string_literal(self, string: str):
        """
        Prints the provided string as an identifier if it is one,
        and as a string literal otherwise.
        """
        if MLIRLexer.bare_identifier_regex.fullmatch(string) is None:
            self.print_string_literal(string)
            return
        self.print_string(string)

    def print_bytes_literal(self, bytestring: bytes):
        self.print_string('"')
        for byte in bytestring:
            match byte:
                case 0x5C:  # ord("\\")
                    self.print_string("\\\\")
                case _ if 0x20 > byte or byte > 0x7E or byte == 0x22:
                    self.print_string(f"\\{byte:02X}")
                case _:
                    self.print_string(chr(byte))
        self.print_string('"')

    def print_float_attr(self, attribute: FloatAttr):
        self.print_float(attribute.value.data, attribute.type)

    def print_complex_float(
        self, value: tuple[float, float], type: ComplexType[ComplexElementCovT]
    ):
        assert isinstance(type.element_type, AnyFloat)
        real, imag = value[0], value[1]
        self.print_string("(")
        self.print_float(real, type.element_type)
        self.print_string(",")
        self.print_float(imag, type.element_type)
        self.print_string(")")

    def print_complex_int(
        self, value: tuple[int, int], type: ComplexType[ComplexElementCovT]
    ):
        assert isinstance(elem_ty := type.element_type, IntegerType)
        real, imag = value[0], value[1]
        if elem_ty.width.data == 1:
            real = "true" if real else "false"
            imag = "true" if imag else "false"
        self.print_string(f"({real},{imag})")

    def print_complex(
        self,
        value: tuple[float, float] | tuple[int, int],
        type: ComplexType[ComplexElementCovT],
    ):
        if isinstance(type.element_type, IntegerType):
            assert isa(value, tuple[int, int])
            self.print_complex_int(value, type)
        else:
            assert isa(value, tuple[float, float])
            self.print_complex_float(value, type)

    def print_float(self, value: float, type: AnyFloat):
        if math.isnan(value) or math.isinf(value):
            if isinstance(type, Float16Type):
                self.print_string(f"{hex(convert_f16_to_u16(value))}")
            elif isinstance(type, Float32Type):
                self.print_string(f"{hex(convert_f32_to_u32(value))}")
            elif isinstance(type, Float64Type):
                self.print_string(f"{hex(convert_f64_to_u64(value))}")
            else:
                raise NotImplementedError(
                    f"Cannot print '{value}' value for float type {str(type)}"
                )
        else:
            # to mirror mlir-opt, attempt to print scientific notation iff the value parses losslessly
            float_str = f"{value:.5e}"
            index = float_str.find("e")
            float_str = float_str[:index] + "0" + float_str[index:]

            parsed_value = type.unpack(type.pack([float(float_str)]), 1)[0]

            if parsed_value == value:
                self.print_string(float_str)
            else:
                if isinstance(type, Float32Type):
                    # f32 is printed with 9 significant digits
                    float_str = f"{value:.9g}"
                    if "." in float_str:
                        self.print_string(float_str)
                    else:
                        self.print_string(f"0x{convert_f32_to_u32(value):X}")
                elif isinstance(type, Float64Type):
                    # f64 is printed with 17 significant digits
                    float_str = f"{value:.17g}"
                    if "." in float_str:
                        self.print_string(float_str)
                    else:
                        self.print_string(f"0x{convert_f64_to_u64(value):X}")
                else:
                    # default to full python precision
                    self.print_string(f"{repr(value)}")

    def print_attribute(self, attribute: Attribute) -> None:
        if isinstance(attribute, UnitAttr):
            self.print_string("unit")
            return

        if isinstance(attribute, LocationAttr):
            self.print_string("loc(unknown)")
            return

        if isinstance(attribute, IntegerType):
            if attribute.signedness.data == Signedness.SIGNLESS:
                self.print_string("i")
            elif attribute.signedness.data == Signedness.SIGNED:
                self.print_string("si")
            elif attribute.signedness.data == Signedness.UNSIGNED:
                self.print_string("ui")
            self.print_string(str(attribute.width.data))
            return

        if isinstance(attribute, BFloat16Type):
            self.print_string("bf16")
            return
        if isinstance(attribute, Float16Type):
            self.print_string("f16")
            return
        if isinstance(attribute, Float32Type):
            self.print_string("f32")
            return
        if isinstance(attribute, Float64Type):
            self.print_string("f64")
            return
        if isinstance(attribute, Float80Type):
            self.print_string("f80")
            return
        if isinstance(attribute, Float128Type):
            self.print_string("f128")
            return

        if isinstance(attribute, IntegerAttr):
            # boolean shorthands
            if (
                isinstance(
                    (ty := attribute.get_type()),
                    IntegerType,
                )
                and ty.width.data == 1
            ):
                self.print_string("true" if attribute.value.data else "false")
                return
            # Otherwise we fall through to TypedAttribute case

        if isinstance(attribute, TypedAttribute) and isinstance(
            attribute, BuiltinAttribute
        ):
            attribute.print_without_type(self)
            self.print_string(" : ")
            self.print_attribute(attribute.get_type())
            return

        if isinstance(attribute, StringAttr):
            self.print_string_literal(attribute.data)
            return

        if isinstance(attribute, BytesAttr):
            self.print_bytes_literal(attribute.data)
            return

        if isinstance(attribute, SymbolRefAttr):
            self.print_string("@")
            self.print_identifier_or_string_literal(attribute.root_reference.data)
            for ref in attribute.nested_references.data:
                self.print_string("::@")
                self.print_identifier_or_string_literal(ref.data)
            return

        # Complex types have MLIR shorthands but XDSL does not.
        if isa(attribute, ComplexType):
            self.print_string("complex<")
            self.print_attribute(attribute.element_type)
            self.print_string(">")
            return

        if isinstance(attribute, ArrayAttr):
            attribute = cast(ArrayAttr[Attribute], attribute)
            self.print_string("[")
            self.print_list(attribute.data, self.print_attribute)
            self.print_string("]")
            return

        if isa(attribute, DenseArrayBase):
            self.print_string("array<")
            self.print_attribute(attribute.elt_type)
            if len(attribute) == 0:
                self.print_string(">")
                return
            data = attribute.iter_values()
            self.print_string(": ")
            # There is a bug in MLIR which will segfault when parsing DenseArrayBase type i1 as 0 or 1,
            # therefore we need to print these as false and true
            if attribute.elt_type == i1:
                self.print_list(
                    data,
                    lambda x: self.print_string("true" if x else "false"),
                )
            else:
                self.print_list(data, lambda x: self.print_string(f"{x}"))
            self.print_string(">")
            return

        if isinstance(attribute, DictionaryAttr):
            self.print_attr_dict(attribute.data)
            return

        if isinstance(attribute, FunctionType):
            self.print_string("(")
            self.print_list(attribute.inputs.data, self.print_attribute)
            self.print_string(") -> ")
            outputs = attribute.outputs.data
            if len(outputs) == 1 and not isinstance(outputs[0], FunctionType):
                self.print_attribute(outputs[0])
            else:
                self.print_string("(")
                self.print_list(outputs, self.print_attribute)
                self.print_string(")")
            return

        if isinstance(attribute, DenseResourceAttr):
            handle = attribute.resource_handle.data
            self.print_string("dense_resource<")
            self.print_resource_handle("builtin", handle)
            self.print_string("> : ")
            self.print_attribute(attribute.type)
            return

        if isinstance(attribute, TensorType):
            attribute = cast(AnyVectorType, attribute)
            self.print_string("tensor<")
            self.print_list(
                attribute.shape.data,
                lambda x: (
                    self.print_string(f"{x.data}")
                    if x.data != -1
                    else self.print_string("?")
                ),
                "x",
            )
            if len(attribute.shape.data) != 0:
                self.print_string("x")
            self.print_attribute(attribute.element_type)
            if isinstance(attribute, TensorType) and attribute.encoding != NoneAttr():
                self.print_string(", ")
                self.print_attribute(attribute.encoding)
            self.print_string(">")
            return

        if isinstance(attribute, VectorType):
            attribute = cast(AnyVectorType, attribute)

            self.print_string("vector<")

            self.print_list(
                zip(attribute.shape, attribute.scalable_dims, strict=True),
                self._print_vector_dim,
                delimiter="x",
            )
            if attribute.shape.data:
                self.print_string("x")

            self.print_attribute(attribute.element_type)
            self.print_string(">")
            return

        if isinstance(attribute, UnrankedTensorType):
            attribute = cast(AnyUnrankedTensorType, attribute)
            self.print_string("tensor<*x")
            self.print_attribute(attribute.element_type)
            self.print_string(">")
            return

        if isinstance(attribute, StridedLayoutAttr):
            self.print_string("strided<[")

            def print_int_or_question(value: IntAttr | NoneAttr) -> None:
                self.print_string(
                    f"{value.data}" if isinstance(value, IntAttr) else "?"
                )

            self.print_list(attribute.strides.data, print_int_or_question, ", ")
            self.print_string("]")
            if attribute.offset == IntAttr(0):
                self.print_string(">")
                return
            self.print_string(", offset: ")
            print_int_or_question(attribute.offset)
            self.print_string(">")
            return

        if isinstance(attribute, MemRefType):
            attribute = cast(MemRefType[Attribute], attribute)
            self.print_string("memref<")
            if attribute.shape.data:
                self.print_list(
                    attribute.shape.data,
                    lambda x: (
                        self.print_string(f"{x.data}")
                        if x.data != -1
                        else self.print_string("?")
                    ),
                    "x",
                )
                self.print_string("x")
            self.print_attribute(attribute.element_type)
            if not isinstance(attribute.layout, NoneAttr):
                self.print_string(", ")
                self.print_attribute(attribute.layout)
            if not isinstance(attribute.memory_space, NoneAttr):
                self.print_string(", ")
                self.print_attribute(attribute.memory_space)
            self.print_string(">")
            return

        if isinstance(attribute, UnrankedMemRefType):
            attribute = cast(AnyUnrankedMemRefType, attribute)
            self.print_string("memref<*x")
            self.print_attribute(attribute.element_type)
            if not isinstance(attribute.memory_space, NoneAttr):
                self.print_string(", ")
                self.print_attribute(attribute.memory_space)
            self.print_string(">")
            return

        if isinstance(attribute, IndexType):
            self.print_string("index")
            return

        if isinstance(attribute, NoneType):
            self.print_string("none")
            return

        if isinstance(attribute, OpaqueAttr):
            self.print_string("opaque<")
            self.print_attribute(attribute.ident)
            self.print_string(", ")
            self.print_attribute(attribute.value)
            self.print_string(">")

            if not isinstance(attribute.type, NoneAttr):
                self.print_string(" : ")
                self.print_attribute(attribute.type)
            return

        if isinstance(attribute, AffineMapAttr):
            self.print_string(f"affine_map<{attribute.data}>")
            return

        if isinstance(attribute, AffineSetAttr):
            self.print_string(f"affine_set<{attribute.data}>")
            return

        if isinstance(attribute, UnregisteredAttr):
            # Do not print `!` or `#` for unregistered builtin attributes
            self.print_string("!" if attribute.is_type.data else "#")
            if attribute.is_opaque.data:
                self.print_string(
                    f"{attribute.attr_name.data.replace('.', '<', 1)}{attribute.value.data}>"
                )
            else:
                self.print_string(attribute.attr_name.data)
                if attribute.value.data:
                    self.print_string(f"<{attribute.value.data}>")
            return

        # Print dialect attributes
        self.print_string("!" if isinstance(attribute, TypeAttribute) else "#")

        if isinstance(attribute, OpaqueSyntaxAttribute):
            self.print_string(attribute.name.replace(".", "<", 1))
            if isinstance(attribute, SpacedOpaqueSyntaxAttribute):
                self.print_string(" ")
        else:
            self.print_string(attribute.name)

        if isinstance(attribute, Data):
            attribute.print_parameter(self)

        elif isinstance(attribute, ParametrizedAttribute):
            attribute.print_parameters(self)

        if isinstance(attribute, OpaqueSyntaxAttribute):
            self.print_string(">")

        return

    def _print_vector_dim(self, pair: tuple[IntAttr, BoolAttr]):
        """
        Helper method to print a vector dimension either as static (`4`) or scalable
        (`[4]`).
        """
        dim, scalable = pair
        if scalable:
            self.print_string(f"[{dim.data}]")
        else:
            self.print_string(f"{dim.data}")

    def print_successors(self, successors: Sequence[Block]):
        if len(successors) == 0:
            return
        self.print_string(" [")
        self.print_list(successors, self.print_block_name)
        self.print_string("]")

    def _print_attr_string(self, attr_tuple: tuple[str, Attribute]) -> None:
        # Print the name without quotes if it is a bare identifier
        if MLIRLexer.bare_identifier_regex.fullmatch(attr_tuple[0]):
            self.print_string(attr_tuple[0])
        else:
            self.print_string(f'"{attr_tuple[0]}"')

        if not isinstance(attr_tuple[1], UnitAttr):
            self.print_string(" = ")
            self.print_attribute(attr_tuple[1])

    def print_attr_dict(self, attr_dict: Mapping[str, Attribute]) -> None:
        self.print_string("{")
        self.print_list(attr_dict.items(), self._print_attr_string)
        self.print_string("}")

    def _print_op_properties(self, properties: dict[str, Attribute]) -> None:
        if not properties:
            return

        self.print_string(" ")
        with self.in_angle_brackets():
            self.print_attr_dict(properties)

    def print_op_attributes(
        self,
        attributes: Mapping[str, Attribute],
        *,
        reserved_attr_names: Iterable[str] = (),
        print_keyword: bool = False,
    ) -> bool:
        """
        Prints the attribute dictionary of an operation, with an optional `attributes`
        keyword.
        Values for `reserved_attr_names` are not printed even if present.
        If the printed dictionary would be empty, then nothing is printed, and this
        function returns False.
        """
        if not attributes:
            return False

        if reserved_attr_names:
            attributes = {
                name: attr
                for name, attr in attributes.items()
                if name not in reserved_attr_names
            }

        if not attributes:
            return False

        if print_keyword:
            self.print_string(" attributes")

        self.print_string(" ")
        self.print_attr_dict(attributes)
        return True

    def print_op_with_default_format(self, op: Operation) -> None:
        self.print_operands(op.operands)
        self.print_successors(op.successors)
        if not self.print_properties_as_attributes:
            self._print_op_properties(op.properties)
        self.print_regions(op.regions)
        if self.print_properties_as_attributes:
            clashing_names = op.properties.keys() & op.attributes.keys()
            if clashing_names:
                raise ValueError(
                    f"Properties {', '.join(clashing_names)} would overwrite the attributes of the same names."
                )

            self.print_op_attributes(op.attributes | op.properties)
        else:
            self.print_op_attributes(op.attributes)
        self.print_string(" : ")
        self.print_operation_type(op)

    def print_function_type(
        self, input_types: Iterable[Attribute], output_types: Iterable[Attribute]
    ):
        """
        Prints a function type like `(i32, i64) -> (f32, f64)` with the following
        format:

        The inputs are always a comma-separated list in parentheses.
        If the output has a single element, the parentheses are dropped, except when the
        only return type is a function type, in which case they are kept.

        ```
        () -> ()                 # no inputs, no outputs
        (i32) -> ()              # one input, no outputs
        (i32) -> i32             # one input, one output
        (i32) -> (i32, i32)      # one input, two outputs
        (i32) -> ((i32) -> i32)  # one input, one function type output
        ```
        """
        self.print_string("(")
        self.print_list(input_types, self.print_attribute)
        self.print_string(") -> ")

        remaining_outputs_iterator = iter(output_types)
        try:
            first_type = next(remaining_outputs_iterator)
        except StopIteration:
            # No outputs
            self.print_string("()")
            return

        try:
            second_type = next(remaining_outputs_iterator)
        except StopIteration:
            # One output, drop parentheses unless it's a FunctionType
            if isinstance(first_type, FunctionType):
                self.print_string("(")
                self.print_attribute(first_type)
                self.print_string(")")
            else:
                self.print_attribute(first_type)
            return

        # Two or more outputs, comma-separated list
        self.print_string("(")
        self.print_list(
            chain((first_type, second_type), remaining_outputs_iterator),
            self.print_attribute,
        )
        self.print_string(")")

    def print_operation_type(self, op: Operation) -> None:
        self.print_function_type(op.operand_types, op.result_types)
        if self.print_debuginfo:
            self.print_string(" loc(unknown)")

    def enter_scope(self) -> None:
        self._next_valid_name_id.append(self._next_valid_name_id[-1])
        self._next_valid_block_id.append(self._next_valid_block_id[-1])
        self._ssa_names.append(self._ssa_names[-1].copy())
        self._block_names.append(self._block_names[-1].copy())

    def exit_scope(self) -> None:
        self._next_valid_name_id.pop()
        self._next_valid_block_id.pop()
        self._ssa_names.pop()
        self._block_names.pop()

    def print_op(self, op: Operation) -> None:
        scope = bool(op.get_traits_of_type(IsolatedFromAbove))
        begin_op_pos = self._current_column
        self._print_results(op)
        if scope:
            self.enter_scope()
        use_custom_format = False
        if isinstance(op, UnregisteredOp):
            self.print_string(f'"{op.op_name.data}"')
        # If we print with the generic format, or the operation does not have a custom
        # format
        elif self.print_generic_format or Operation.print is type(op).print:
            self.print_string(f'"{op.name}"')
        else:
            self.print_string(f"{op.name}")
            use_custom_format = True
        end_op_pos = self._current_column
        if op in self.diagnostic.op_messages:
            for message in self.diagnostic.op_messages[op]:
                self._add_message_on_next_line(message, begin_op_pos, end_op_pos)
        if isinstance(op, UnregisteredOp):
            op_name = op.op_name
            del op.attributes["op_name__"]
            self.print_op_with_default_format(op)
            op.attributes["op_name__"] = op_name
        elif use_custom_format:
            op.print(self)
        else:
            self.print_op_with_default_format(op)
        if scope:
            self.exit_scope()

    def print_resource_handle(self, dialect: str, handle: str) -> None:
        if dialect not in self._dialect_resources:
            self._dialect_resources[dialect] = set()
        self._dialect_resources[dialect].add(handle)
        self.print_string(handle)

    def print_metadata(self, dialects: Iterable[Dialect]) -> None:
        if not self._dialect_resources:
            return

        # Prepare data
        resources_for_printing: dict[str, dict[str, str]] = {}
        resource_dialects = {
            d.name: d.get_interface(OpAsmDialectInterface)
            for d in dialects
            if d.has_interface(OpAsmDialectInterface)
        }

        for dialect_name, resource_keys in self._dialect_resources.items():
            interface = resource_dialects.get(dialect_name)
            assert interface
            resources = interface.build_resources(resource_keys)
            if resources:
                resources_for_printing[dialect_name] = resources

        if not resources_for_printing:
            # None of the referenced resources actually exist
            return

        # Printing
        self.print_string("\n\n{-#\n")
        self.print_string("  dialect_resources: {\n")

        for dialect_name, resources in resources_for_printing.items():
            self.print_string("    " + dialect_name + ": {\n")
            sorted_elements = sorted(resources.items(), key=lambda x: x[0])
            for key, resource in sorted_elements[:-1]:
                self.print_string(f'      {key}: "{resource}",')
            self.print_string(
                f'      {sorted_elements[-1][0]}: "{sorted_elements[-1][1]}"'
            )
            self.print_string("\n    }\n")

        self.print_string("  }\n")
        self.print_string("#-}")
