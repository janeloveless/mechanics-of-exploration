#! /usr/bin/env python

import sympy as sy

t = sy.symbols("t")
V = sy.symbols("V")
I = sy.symbols("I")

rv_i = 0
def unique_rv(root="rv") :
    global rv_i
    rv = sy.stats.Uniform(str(root) + "_" + str(rv_i), 0, 1)
    rv_i = rv_i + 1
    return rv
