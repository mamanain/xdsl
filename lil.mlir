%v_f32, %v_f64, %v_i32 = "test.op"() : () -> (f32, f64, i32)
%r, %c = "test.op"() : () -> (index, index)
%m_f32, %m_f64, %m_i32, %m_scalar_i32 = "test.op"() : () -> (memref<3x2xf32>, memref<3x2xf64>, memref<3xi32>, memref<i32>)
memref.store %v_f32, %m_f32[%r, %c] {"nontemporal" = false} : memref<3x2xf32>
