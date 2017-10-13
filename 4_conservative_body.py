#! /usr/bin/env python2

import itertools as it

import numpy as np
import scipy as sp
import scipy.linalg
import sympy as sy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import neuromech as nm
from neuromech.symbol import t

"""
Simulate and analyse the motion of the body at large amplitudes under the
assumption of energy conservation (no friction or driving).
"""

"""
Set some parameters. 

Some others will be defined later in the code, where it is more convenient.
"""

print "Defining parameters..."

# mechanical parameters
N = 12                                              # number of segments
k_a = sy.symbols("k_a", real=True, positive=True)   # uniform axial stiffness parameter
k_t = sy.symbols("k_t", real=True, positive=True)   # uniform transverse stiffness parameter
k_f = sy.symbols("k_f", real=True, positive=True)   # fluid stiffness

# filesystem parameters
PLOT_PATH = "./data/output/4_conservative_body/"
F_PATH = "./FORTRAN_sources/"

# plotting parameters
fontsize = 12
output_dpi = 450
SAVE_PLOTS = True
SHOW_PLOTS = True
 

"""
Construct mechanical system.
"""

print "Building model of conservative body motion..."
model = nm.model.SimplePlanarCrawler(N, 
        k_axial=k_a, 
        k_lateral=[k_t]*(N - 2), 
        k_fluid=k_f,
        n_axial=0, 
        n_lateral=[0]*(N - 2), 
        mu_f=[0]*N, 
        mu_b=[0]*N,
        b=[0]*(N - 1), 
        c=[0]*(N - 2))

f = model.f                                         # dynamical equations
x = model.x                                         # state vector
params = model.parameters                           # parameter vector
H = model.H                                         # Hamiltonian


"""
Compile equations of motion to FORTRAN, then to binary.
"""

print "Compiling RHS function to intermediate FORTRAN source code..."
f_src = model.FORTRAN_f(verbose=True)

# save FORTRAN source code for future usage -- if code above this line changes
with open(F_PATH + "4_conservative_body.f", "w") as src_file :
    src_file.write(f_src)

# load FORTRAN source code
f_src = open(F_PATH + "4_conservative_body.f").read()

print "Compiling RHS function FORTRAN source code to binary..."
f_f77 = nm.util.FORTRAN_compile(f_src)


"""
Set initial conditions.
"""

print "Setting simulation parameters and initial conditions..."

IC_PATH = "./data/initial_conditions/4_conservative_body/"
# set mechanical initial conditions
# ... first load mechanical mode shapes
v_a = np.load(IC_PATH + "axial_modal_ics.npy")            # load axial mode shapes
v_t = np.load(IC_PATH + "transverse_modal_ics.npy")       # load transverse mode shapes

conf_scale = 1.

# ... initialise mechanical state vector to zero, then construct a starting
# state vector from low frequency mode shapes (ignoring total translations and
# rotations)
x0 = np.zeros(4*N)                   
x0[:2*N:2] = conf_scale*np.append(v_a[2], v_a[2][0]) + np.arange(N)     # x
x0[1:2*N:2] = conf_scale*(v_t[2] - v_t[3] - v_t[4])                     # y
orig_x0 = np.copy(x0)   # store a copy of this IC, before adding noise

# ... then specify distance between starting IC and "noised" IC
epsilon = 0.0000001

# ... add position noise
x0[:2*N:2] = x0[:2*N:2] + epsilon*2*(np.random.random(N) - 0.5)         # x
x0[1:2*N:2] = x0[1:2*N:2] + epsilon*2*(np.random.random(N) - 0.5)       # y

# ... add momentum noise
x0[2*N:][0::2] = x0[2*N:][0::2] + epsilon*2*(np.random.random(N) - 0.5) # x
x0[2*N:][1::2] = x0[2*N:][1::2] + epsilon*2*(np.random.random(N) - 0.5) # y

# ... remove centre of mass momentum
x0[2*N:][0::2] = x0[2*N:][0::2] - np.mean(x0[2*N:][0::2])               # x
x0[2*N:][1::2] = x0[2*N:][1::2] - np.mean(x0[2*N:][1::2])               # y

# find total length of larva, given initial conditions
L0 = np.sum(np.linalg.norm(np.diff(x0[:2*N].reshape(-1, 2), axis=0), axis=1))

# set neural state to zero and combine with mechanical initial conditions

print str(len(x0)) + " initial conditions have been set."


t_arr = np.linspace(0, 400, 200000)    # simulation time axis

p0 = [L0, (2*np.pi)**2, 1000] +\
     [(2*np.pi*np.exp(1)/6.)**2]*10 + \
     [1, 1]
p0 = [L0,                               # total length
      (2*np.pi)**2,                     # axial stiffness
      1000,                             # fluid stiffness
      (2*np.pi*np.exp(1)/6.)**2,        # transverse stiffness
      1,                                # segment length
      1]                                # segment mass

tol = 10**-12

print str(len(p0)) + " free parameters have been set."


"""
Define numerical energy function.
"""

print "Defining numerical energy function..."
H_lam = sy.lambdify([t] + x, H.subs(zip(params, p0)))
H_num = lambda x : np.array(H_lam(0, *x), dtype=np.float).flatten()


"""
Simulate, analyse output, then plot.
"""

# run simulation

print "Attempting simulation run..."
if len(params) == len(p0) and len(x) == len(x0) :
    x_arr = nm.util.FORTRAN_integrate(t_arr, x0, f_f77, p0, rtol=tol, atol=tol)
else :
    raise Exception("length mismatch in parameter or IC vector")
print "Simulation completed successfully!"

print "Computing segment lengths and bending angles..."
q_vec_arr = x_arr[:, :2*N].reshape(len(t_arr), -1, 2)
length_arr = np.linalg.norm(np.diff(q_vec_arr, axis=1), axis=2)
angles_arr = np.diff(np.arctan2(np.diff(q_vec_arr, axis=1)[:, :, 0],
                                np.diff(q_vec_arr, axis=1)[:, :, 1]), axis=1)

print "Calculating power spectra..."
psd_q = nm.analysis.psd(t_arr, length_arr[:, -1], timescale=1)
psd_phi = nm.analysis.psd(t_arr, angles_arr[:, -1], timescale=1)

print "Calculating autocorrelation..."
corr_q = nm.analysis.correlation(t_arr, length_arr[:, -1] - np.mean(length_arr[:, -1]), 
                                        length_arr[:, -1] - np.mean(length_arr[:, -1]))
corr_phi = nm.analysis.correlation(t_arr, angles_arr[:, -1] - np.mean(angles_arr[:, -1]), 
                                          angles_arr[:, -1] - np.mean(angles_arr[:, -1]))

print "Attempting to estimate maximum Lyapunov characteristic exponent..."
lce_analysis = nm.analysis.lce_estimate(x0, f_f77, p0, t_step=(t_arr[1] -
    t_arr[0])/200., pb_step=2000, n_pb=2010, n_ic_steps=4,
    n_pb_discard=10, log=np.log2, tol=tol, debug=False, E=H_num)

print "Plotting results..."
plt.ioff()
fig = plt.figure("kinematics", figsize=(2.5, 5))
plt.clf()
plot = fig.add_subplot(111)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.plot(x_arr[:, 1:2*N:2], x_arr[:, :2*N:2], c='k', lw=0.05)
plt.xlim(-4, 4)
plt.xticks([-4, -2, 0, 2, 4])
plt.ylim(-2, 14)
plt.xlabel("x displacement", fontsize=fontsize)
plt.ylabel("y displacement", fontsize=fontsize)
plt.grid(False)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "kinematics.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close() 
plt.ion()
plt.show()

plt.ioff()
fig = plt.figure("chaos analysis", figsize=(5, 5))
plt.clf()
plt.subplot(323)
plt.plot(psd_q[0], np.log(psd_q[1]), c='k')
plt.xlim(0, 2)
plt.ylim(-5, 20)
plt.ylabel("log PSD $q$")
plt.grid(False)

plt.subplot(325)
plt.cla()
plt.plot(psd_phi[0], np.log(psd_phi[1]), c='k')
plt.xlim(0, 2)
plt.ylim(-5, 20)
plt.ylabel("log PSD $\phi$")
plt.xlabel("frequency (Hz)")
plt.grid(False)

plt.subplot(324)
plt.cla()
plt.plot(corr_q[0], corr_q[1]/np.max(corr_q[1]), c='k')
plt.xlim(corr_q[0][0], corr_q[0][-1])
plt.ylim(-1, 1)
plt.ylabel("autocorr. $q$")
plt.grid(False)

plt.subplot(326)
plt.plot(corr_phi[0], corr_phi[1]/np.max(corr_phi[1]), c='k')
plt.xlim(corr_phi[0][0], corr_phi[0][-1])
plt.ylim(-1, 1)
plt.ylabel("autocorr. $\phi$")
plt.xlabel("time lag (s)")
plt.grid(False)

plt.subplot(321)
plt.cla()
plt.plot(lce_analysis[1], c='k')
plt.xlim(0, len(lce_analysis[1]))
plt.ylim(-0.5, 2)
plt.ylabel("MLCE (bits s$^{-1}$)")
plt.axhline(0, c='gray', alpha=0.5, lw=2)
plt.grid(False)

#plt.subplot(322)
#plt.cla()
#plt.plot(lce_analysis[2], c='k')
#plt.xlim(0, len(lce_analysis[2]))
#plt.ylim(-70, 70)
#plt.yticks([-70, -35, 0, 35, 70])
#plt.ylabel("FT-LCE (bits s$^{-1}$)")
#plt.xlabel("iteration")
#plt.grid(False)

nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "chaos_analysis.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()

