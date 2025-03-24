%c0, %end, %c1, %mult, %shift = "test.op"() : () -> (index, index, index, index, index)
scf.for %i = %c0 to %end step %c1 {
        scf.for %j = %c0 to %end step %c1 {
		%a = arith.muli %i, %mult : index
		%b = arith.addi %a, %j : index
		%c = arith.muli %b, %mult : index
		"test.op"(%c) : (index) -> ()
	}
}
