%shift, %mul_shift, %lb, %ub, %step = "test.op"() : () -> (index, index, index, index, index)

scf.for %i = %lb to %ub step %step {
      scf.for %j = %lb to %ub step %step {
	%a = arith.muli %i, %mul_shift : index
        %b = arith.addi %a, %j : index
	%c = arith.muli %b, %mul_shift : index
	"test.op"(%b) : (index) -> () 
      }
    }
