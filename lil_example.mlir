"func.func"() <{sym_name = "fill2d", function_type = (memref<10x10xi32>) -> ()}> ({
  ^0(%m : memref<10x10xi32>):
    %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %end = "arith.constant"() <{value = 10 : index}> : () -> index
    %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %val = "arith.constant"() <{value = 100 : i32}> : () -> i32
    %pointer_dim_stride = "arith.constant"() <{value = 10 : index}> : () -> index
    %bytes_per_element = "arith.constant"() <{value = 4 : index}> : () -> index
    "scf.for"(%c0, %end, %c1) ({
    ^1(%i : index):
      "scf.for"(%c0, %end, %c1) ({
      ^2(%j : index):
        %pointer_dim_offset = "arith.muli"(%i, %pointer_dim_stride) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %pointer_dim_stride_1 = "arith.addi"(%pointer_dim_offset, %j) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %scaled_pointer_offset = "arith.muli"(%pointer_dim_stride_1, %bytes_per_element) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %0 = "ptr_xdsl.to_ptr"(%m) : (memref<10x10xi32>) -> !ptr_xdsl.ptr
        %offset_pointer = "ptr_xdsl.ptradd"(%0, %scaled_pointer_offset) : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
        "ptr_xdsl.store"(%offset_pointer, %val) : (!ptr_xdsl.ptr, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
