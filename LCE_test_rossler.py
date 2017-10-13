#! /usr/bin/env python2

import sympy as sy
import sympy.physics.mechanics as mech
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import neuromech as nm


"""
In this script we analyse the 3D Rossler system (a classic example of chaotic
behaviour) using a numerical estimate of maximal Lyapunov characteristic
exponent.
"""

"""
Define variables and parameters.
"""

x, y, z = mech.dynamicsymbols("x, y, z")    # state variables

u = sy.Matrix([x, y, z])                       # state vector
p = []                                            # there is no parameter vector


"""
Construct dynamical equations.
"""

fx = -y -z
fy = x + 0.2*y
fz = 0.2 + z*(x - 5.7)
f = sy.Matrix([fx, fy, fz])


"""
Construct the Jacobian of the dynamics.
"""

#J = f.jacobian(w)


"""
Compile the dynamics and Jacobian to FORTRAN, then to binary for fast numerical
evaluation.
"""

f_src = nm.util.FORTRAN_f(u, f, p)
#J_src = nm.util.FORTRAN_jacobian(w, J, p)
f_f77 = nm.util.FORTRAN_compile(f_src)


"""
Evaluate the system numerically using the legacy FORTRAN LSODEs solver.
"""

# simulation tolerance, time axis, and numerical parameters / initial conditions
tol = 10**-13

t0 = 0
tF = 200
dt = 0.01
t_arr = np.linspace(t0, tF, (tF - t0)/dt)

u0 = [0, 0, 0]
p0 = []

up_arr = nm.util.FORTRAN_integrate(t_arr, u0, f_f77, p0)
u_arr = up_arr[:, :3]
p_arr = up_arr[:, 3:]


"""
Numerically estimate the maximal Lyapunov characteristic exponent.
"""

lce_accepted = 0.07
lce_estimate = nm.analysis.lce_estimate(u_arr[-1], f_f77, p0, t_step=dt,
                                n_pb=20000, pb_step=100, n_ic_steps=2, tol=tol)
print "maximal Lyapunov characteristic exponent estimate = %0.3f (accepted = %0.2f)" % (lce_estimate[0][0], lce_accepted)

