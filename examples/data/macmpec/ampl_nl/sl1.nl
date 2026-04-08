g3 0 1 0	# problem sl1
 8 5 1 0 2	# vars, constraints, objectives, ranges, eqns
 0 1 3 0 0 0	# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb
 0 0	# network constraints: nonlinear, linear
 0 2 0	# nonlinear vars in constraints, objectives, both
 0 0 0 1	# linear network variables; functions; arith, flags
 0 0 0 0 0	# discrete variables: binary, integer, nonlinear (b,c,o)
 13 2	# nonzeros in Jacobian, gradients
 0 0	# max name lengths: constraints, variables
 0 0 0 0 0	# common exprs: b,c,o,c1,o1
b
3
3
2 10
0 0.01 10
0 0 1
2 0
2 0
2 0
r
4 0
4 0
5 1 6
5 1 7
5 1 8
C0
n0
C1
n0
C2
n-10
C3
n-2
C4
n0
O0 0
o0
o5
o0
n-2
v0
n2
o5
v1
n2
k7
3
6
7
8
9
11
12
J0 3
0 0.02
5 -10
6 -1
J1 3
1 2
5 -1
7 -1
J2 3
0 10
1 1
2 -1
J3 2
0 1
3 -1
J4 2
1 1
4 -50
G0 2
0 0
1 0
