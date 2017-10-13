#! /usr/bin/env python2

import sympy as sy
import sympy.physics.mechanics as mech
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import neuromech as nm


"""
In this script we analyse the Lorenz system (a classic example of chaotic
behaviour) using some analytical tools and numerical estimates ofmaximal
Lyapunov characteristic exponent and fractal dimension.
"""

"""
Define variables and parameters.
"""

x, y, z = mech.dynamicsymbols("x, y, z")    # state variables
s, r, b = sy.symbols("s, r, b")             # Prandtl number, Rayleigh number,
                                            # and an unnamed parameter

w = sy.Matrix([x, y, z])                    # state vector
p = [s, r, b]                               # parameter vector


"""
Construct dynamical equations.
"""

fx = s*(y - x)
fy = r*x - y - x*z
fz = x*y - b*z
f = sy.Matrix([fx, fy, fz])


"""
Construct the Jacobian of the dynamics.
"""

#J = f.jacobian(w)


"""
Apply analytical tools to find fixed points and analyse their stability using
the eigenvalues and eigenvectors of the Jacobian.
"""
# TODO -- I have worked this by hand in my notebook so it's non-urgent

"""
Compile the dynamics and Jacobian to FORTRAN, then to binary for fast numerical
evaluation.
"""

f_src = nm.util.FORTRAN_f(w, f, p)
#J_src = nm.util.FORTRAN_jacobian(w, J, p)
f_f77 = nm.util.FORTRAN_compile(f_src)


"""
Evaluate the system numerically using the legacy FORTRAN LSODEs solver.
"""

# simulation tolerance, time axis, and numerical parameters / initial conditions
tol = 0.0001

t0 = 0
tF = 5000
dt = 0.25
t_arr = np.linspace(t0, tF, (tF - t0)/dt)

p0 = [10, 28, 8./3.]
w0 = [0, 1, 0]

wp_arr = nm.util.FORTRAN_integrate(t_arr, w0, f_f77, p0)
w_arr = wp_arr[:, :3]
p_arr = wp_arr[:, 3:]


"""
Numerically estimate the dimension of the solution using the correlation
dimension. The accepted estimate is ~2.05 according to Strogatz' "Nonlinear
Dynamic and Chaos".
"""

D_accepted = 2.05
D_estimate = nm.analysis.correlation_dimension_estimate(w_arr[len(w_arr)/2:], debug=False)
print "correlation dimension estimate = %0.2f (accepted = %0.2f)" % (D_estimate, D_accepted)


"""
Numerically estimate the maximal Lyapunov characteristic exponent. The accepted
estimate is ~0.9 according to Strogatz' "Nonlinear Dynamics and Chaos".
"""

lce_accepted = 0.9
lce_estimate = nm.analysis.lce_estimate(w0, f_f77, p0, t_step=0.0001,
        n_ic_steps=10000, n_pb=20000, pb_step=500, tol=10**-13,
        n_pb_discard=100, debug=False)
print "maximal Lyapunov characteristic exponent estimate = %0.2f (accepted = %0.1f)" % (lce_estimate[0][0], lce_accepted)
