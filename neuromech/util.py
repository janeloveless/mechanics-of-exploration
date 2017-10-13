#! /usr/bin/env python

import os
import itertools as it
import sys
import textwrap
#import gtk
import numpy as np
import sympy as sy
import sympy.stats
import odespy as ode
import matplotlib
import matplotlib.pyplot as plt
import sympy.physics.mechanics as mech


"""
Pretty plotting code.
"""

_all_spines = ["top", "right", "bottom", "left"]
def hide_spines(s=["top", "right"]):
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    global _all_spines

    # Retrieve a list of all current figures.
    figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            for spine in _all_spines :
                if spine in s :
                    ax.spines[spine].set_color('none')

                if "top" in s and "bottom" in s :
                    ax.xaxis.set_ticks_position('none')
                elif "top" in s :
                    ax.xaxis.set_ticks_position('bottom')
                elif "bottom" in s :
                    ax.xaxis.set_ticks_position('top')
                else :
                    ax.xaxis.set_ticks_position('both')

                if "left" in s and "right" in s :
                    ax.yaxis.set_ticks_position('none')
                elif "left" in s :
                    ax.yaxis.set_ticks_position('right')
                elif "right" in s :
                    ax.yaxis.set_ticks_position('left')
                else : 
                    ax.yaxis.set_ticks_position('both')


"""
FORTRAN compilation code.
"""

def find_matching_parentheses(s, popen="(", pclose=")") :
    i_start = s.find(popen)

    i_end = -1
    count = 0
    s_frame = s[i_start:]
    for i in xrange(len(s_frame)) :
        char = s_frame[i]
        if char == popen :
            count += 1
        elif char == pclose :
            count -= 1
        if count == 0 :
            i_end = i + i_start + 1
            break
    return i_start, i_end


def parse_merge(H, s) :
    """
    Parse the first FORTRAN merge statement found within s.
    H is the name of a hidden variable which will be used to store the value of
    the piecewise function defined by the merge statement.
    """
    # extract bracketed code in merge statement from s
    # m_statement is of form "(expr1,expr2,cond)"
    i_merge_start = s.find("merge")
    ms = s[i_merge_start:]
    i_start, i_end = find_matching_parentheses(ms)
    m_statement = ms[i_start:i_end]

#    print m_statement
    
    # extract expr1, expr2, and conditional
    i1 = m_statement.find(",")
    i2 = m_statement.rfind(",")
    expr1 = m_statement[1:i1]
    expr2 = m_statement[i1 + 1:i2]
    cond = m_statement[i2 + 1:-1]

    # if expr1, expr2, or cond are merge statements, recursively call this
    # function otherwise, set the hidden switch variable to take the value of
    # the relevant expr
    if expr1.find("merge") != -1 :
        expr1_str = parse_merge(H, expr1)[-1]
        expr1_str = "".join(["      " + s + "\n" for s in expr1_str.splitlines()])
    else :
        expr1_str = "            " + H + "=" + expr1

    if expr2.find("merge") != -1 :
        expr2_str = parse_merge(H, expr2)[-1]
        expr2_str = "".join(["      " + s + "\n" for s in expr2_str.splitlines()])
    else :
        expr2_str = "            " + H + "=" + expr2

    # format expr1_str, expr2_str, and cond into a correct FORTRAN IF-THEN-ELSE
    # statement
    f_code = "      IF (" + cond.strip() + ") THEN \n" + expr1_str + "\n" + \
             "      ELSE \n" + expr2_str + "\n" + \
             "      ENDIF \n"

    return i_merge_start, i_merge_start + i_end, f_code


def FORTRAN_f(x, f, parameters=[], verbose=False) :
    """
    Produce FORTRAN function for evaluating a vector-valued SymPy expression f
    given a state vector x. 
    
    The FORTRAN function will have the signature f_f77(neq, t, X, Y) where neq
    is hidden and Y is an output matrix.
    """
    # TODO remove code for dealing with stochastic systems -- it is not used in
    # this paper
    x = list(x) + list(parameters)
    f = list(f) + [0]*len(parameters)
    rv = list(set((np.concatenate([sy.stats.random_symbols(f_i) for f_i in f]))))

    NR = len(rv)
    if NR > 0 :
        x += [sy.symbols("dt"), sy.symbols("seed")]
        f += [0, 0]
    NX = len(x)
    NY = len(f)
    if NX != NY :
        raise Exception("System is not square!")

    if verbose : print "generating FORTRAN matrices..."
    _X = sy.tensor.IndexedBase("X", shape=(NX, ))
    X = [_X[i + 1] for i in xrange(NX)]

    _R = sy.tensor.IndexedBase("R", shape=(NR, ))
    R = [_R[i + 1] for i in xrange(NR)]
    
    if type(f) != sy.Matrix : f = sy.Matrix(f)
    # WARNING : These substitution steps are VERY SLOW!!! It might be wise to
    # parallelise them in the future, or at least substitute into one dynamical
    # equation at a time so that progress can be monitored.
    if verbose : print "substituting matrix elements for original state variables and parameters (WARNING: SLOW)..."
    f_sub = f.subs(zip(x, X))                   
    if verbose : print "substituting matrix elements for random variables (WARNING: SLOW)..."
    f_sub = f_sub.subs(zip(rv, R))

    # generate FORTRAN code
    if verbose : print "generating FORTRAN code from dynamics equations..."
    fstrs = [sy.fcode(fi, standard=95) for fi in f_sub]

    # remove whitespace and newlines
    if verbose : print "removing whitespace and newlines..."
    fstrs = ["".join(fi.split()) for fi in fstrs]

    # remove all @ (FORTRAN line continuation indicator)
    if verbose : print "removing line continuations..."
    fstrs = [fi.replace("@", "") for fi in fstrs]

    # find FORTRAN inline merge statements and replace with a hidden "switch"
    # variable whose value is set by a full IF statement at the start of the
    # function call.
    # -- this is needed because FORTRAN77 doesn't support inline merge statements
    Hstrs = []  # to hold hidden switch expressions

    if verbose : print "formatting piecewise functions..."
    for i in xrange(len(fstrs)) :
        while fstrs[i].find("merge") != -1 :
            H = "H(" + str(len(Hstrs) + 1) + ")"
            i_merge_start, i_merge_end, Hstr = parse_merge(H, fstrs[i])
            fstrs[i] = fstrs[i][:i_merge_start] + H + fstrs[i][i_merge_end:]
            Hstrs.append(Hstr)
    NH = len(Hstrs)

    # format the fstrs
    wrapper = textwrap.TextWrapper(expand_tabs=True, 
                                   replace_whitespace=True,
                                   initial_indent="      ", 
                                   subsequent_indent="     @    ", 
                                   width=60)

    if verbose : print "formatting state equations..."
    for i in xrange(len(fstrs)) :
        fstrs[i] = wrapper.fill("Y(" + str(i + 1) + ")=" + fstrs[i]) + "\n"

    # put the above elements together into a FORTRAN subroutine
    if verbose : print "formatting preamble..."
    hdr = "      subroutine f_f77(neq, t, X, Y) \n" +\
          "Cf2py intent(hide) neq \n" +\
          "Cf2py intent(out) Y \n" +\
          "      integer neq \n" +\
          "      double precision t, X, Y \n" +\
          "      dimension X(neq), Y(neq) \n"
    if NH > 0 : hdr += "      real, dimension(" + str(NH) + ") :: H \n"
    # TODO fix the following -- assumes dt = 0.01
    # NOTE this is only important when dealing with stochastic systems
    if NR > 0 : hdr += "      real, dimension(" + str(NR) + ") :: R \n" +\
                       "      integer :: SEED \n" +\
                       "      real :: RTRASH \n" +\
                       "      SEED = INT((t/" + sy.fcode(X[-2]).strip() +\
                                ") + " + sy.fcode(X[-1]).strip() + ") \n" +\
                       "      CALL SRAND(SEED) \n" +\
                       "      DO i=1,4 \n" +\
                       "            RTRASH=RAND(0) \n" +\
                       "      END DO \n"
    R_block = "".join([sy.fcode(R_i) + "=RAND(0) \n" for R_i in R])
    H_block = "".join(Hstrs)
    Y_block = "".join(fstrs)

    if verbose : print "assembling source code blocks..."
    fcode = hdr + R_block + H_block + Y_block + "      return \n" + "      end \n"

    # final formatting
    if verbose : print "final source code formatting..."
    wrapper = textwrap.TextWrapper(expand_tabs=True, replace_whitespace=True,
                                   initial_indent="", subsequent_indent="     @    ", width=60)

    fcode = "".join([wrapper.fill(src) + "\n" for src in fcode.split("\n")])

    return fcode


def FORTRAN_jacobian(x, jac, parameters=[]) :
    # TODO document
    # TODO remove this function if unused in paper
    NX = len(x)
    NP = len(parameters)
    Nrowpd = jac.shape[0]
    Ncolpd = jac.shape[1]
    if NX != Nrowpd != Ncolpd :
        raise Exception("System is not square!")


    _X = sy.tensor.IndexedBase("X", shape=(NX, ))
    X = [_X[i + 1] for i in xrange(NX)]
    X = X + [_X[NX + i + 1] for i in xrange(NP)]

    if type(jac) == sy.Matrix : jac = sy.Matrix(jac)
    jac_sub = jac.subs(zip(list(x) + list(parameters), X))

    ijs = [i for i in it.product(xrange(Nrowpd), xrange(Ncolpd))]

    # generate FORTRAN code
    fstrs = [sy.fcode(jac_ij) for jac_ij in jac_sub]

    # remove whitespace and newlines
    fstrs = ["".join(jac_ij.split()) for jac_ij in fstrs]

    # remove all @ (FORTRAN line continuation indicator)
    fstrs = [jac_ij.replace("@", "") for jac_ij in fstrs]

    # find FORTRAN inline merge statements and replace with a hidden "switch"
    # variable whose value is set by a full IF statement at the start of the
    # function call.
    # -- this is needed because FORTRAN77 doesn't support inline merge statements
    Hstrs = []  # to hold hidden switch expressions

    for i in xrange(len(fstrs)) :
        while fstrs[i].find("merge") != -1 :
            H = "H(" + str(len(Hstrs) + 1) + ")"
            i_merge_start, i_merge_end, Hstr = parse_merge(H, fstrs[i])
            fstrs[i] = fstrs[i][:i_merge_start] + H + fstrs[i][i_merge_end:]
            Hstrs.append(Hstr)
    NH = len(Hstrs)

    # format the fstrs
    wrapper = textwrap.TextWrapper(expand_tabs=True, 
                                   replace_whitespace=True,
                                   initial_indent="      ", 
                                   subsequent_indent="     @    ", 
                                   width=60)

    for k in xrange(len(fstrs)) :
        i, j = ijs[k]
        fstrs[k] = wrapper.fill("pd(" + str(i + 1) + "," + str(j + 1) + ")=" + fstrs[k]) + "\n"

    # put the above elements together into a FORTRAN subroutine
    hdr = "      subroutine jac_f77(neq, t, X, ml, mu, pd, nrowpd) \n" +\
          "Cf2py intent(hide) neq, ml, mu, nrowpd \n" +\
          "Cf2py intent(out) pd \n" +\
          "      integer neq, ml, mu, nrowpd \n" +\
          "      double precision t, X, pd \n" +\
          "      dimension X(neq), pd(neq, neq) \n"
    if NH > 0 : hdr += "      real, dimension(" + str(NH) + ") :: H \n"
    H_block = "".join(Hstrs)
    pd_block = "".join(fstrs)

    fcode = hdr + H_block + pd_block + "      return \n" + "      end \n"

    return fcode


def FORTRAN_compile(fcode) :
    f_f77 = ode.compile_f77(fcode)
    os.remove("tmp_callback.so")
#    reload(ode)
    return f_f77


"""
Numerical integration code.
"""

def FORTRAN_integrate(t, x0, f, p0=[], jac=None, rtol=0.0001, atol=0.0001) :
    solver = ode.Lsodes(f=None, f_f77=f, jac_f77=jac, rtol=rtol, atol=atol)
    solver.set_initial_condition(list(x0) + list(p0))
    x, _ = solver.solve(t)
    return x
