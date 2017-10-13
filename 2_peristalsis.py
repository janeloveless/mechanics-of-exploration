#! /usr/bin/env python2

import itertools as it

import numpy as np
import scipy as sp
import scipy.linalg
import sympy as sy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import neuromech as nm


"""
Simulate and analyse axial motion in the presence of friction and driving
forces.
"""

"""
Set some parameters. 

Some others will be defined later in the code, where it is more convenient.
"""

print "Setting parameters..."

# mechanical parameters
N_seg = 11                              # number of segments in crawler model
mech_timescale = 1./1000.               # timescale of mechanical system, relative to neural system

# filesystem parameters
PLOT_PATH = "./data/output/2_peristalsis/plots/"
F_PATH = "./FORTRAN_sources/"

# plotting parameters
SAVE_PLOTS = True
SHOW_PLOTS = True
fontsize=12
output_dpi=450


"""
Set up the mechanical system.
"""

print "Setting up mechanical system..."
m = nm.model.NondimensionalHarmonicCrawler(N_seg)


"""
Set up the neural system.
"""

print "Setting up neural system..."

print "Setting sensory neuron inputs to mechanical outputs..."
D1 = -sy.Matrix(sp.linalg.circulant([-1, 1] + [0]*(N_seg - 2))) # circulant first difference matrix

SN_q_gain = 0                                               # SN position weight
SN_p_gain = 1                                               # SN momentum weight

SN_u = m.x                                                  # SN sensitive to mech state
SN_q_ws = (SN_q_gain*D1.T).tolist()                         # stretch input weight
SN_p_ws = (SN_p_gain*D1.T).tolist()                         # stretch rate input weight
SN_ws = [q_w + p_w for q_w, p_w in zip(SN_q_ws, SN_p_ws)]   # combine weights
 
n = nm.model.MechanicalFeedbackAndMutualInhibition(N_seg, SN_u, SN_ws)

print "Setting mechanical inputs to motor neuron outputs..."
V_MNs = n.x[2*N_seg:]                                       # motor neuron activations
m.f = m.f.subs(zip(m.u, n.x[2*N_seg:]))                     # substitute into mech eqns
 

"""
Fetch and combine differential equations for each subsystem.
"""

print "Combining dynamical equations, collecting parameters..."
f = sy.Matrix(list(n.f) + list(m.f*mech_timescale))
x = sy.Matrix(list(n.x) + list(m.x))
model = nm.model.DynamicalModel(x=x, f=f)
params = model.parameters


"""
Compile symbolic dynamical equations to FORTRAN, then to binary for efficient
simulation.
"""

print "Compiling RHS function to intermediate FORTRAN source code..."
f_src = model.FORTRAN_f(verbose=True)

# save FORTRAN source code for future usage
with open(F_PATH + "2_peristalsis.f", "w") as src_file :
    src_file.write(f_src)

# load FORTRAN source code
f_src = open(F_PATH + "2_peristalsis.f").read()

print "Compiling RHS function FORTRAN source code to binary..."
f_f77 = nm.util.FORTRAN_compile(f_src)


"""
Define simulation parameters.
"""

print "Setting simulation parameters..."
tol = 0.0006                            # absolute and relative tolerance for numerical integrator
t = np.linspace(0, 24000, 72000)        # simulation time axis (long)

print "Setting representative model parameters..."
p0 = [-2, -2, 4, 1, 1, 2*np.pi, 0.01, 0.5]

"""
Plot forward and backward soliton.
"""

print "Generating forward and backward waves..."

# load initial conditions leading to forward and backward waves
print "Loading initial conditions..."
IC_PATH = "./data/initial_conditions/2_peristalsis/"
x0_fw = np.load(IC_PATH + "forward_wave_ic.npy")
x0_bk = np.load(IC_PATH + "backward_wave_ic.npy")

# integrate from initial conditions
print "Numerically integrating..."
x_arr_fw = nm.util.FORTRAN_integrate(t, x0_fw, f_f77, p0, rtol=tol, atol=tol)
x_arr_bk = nm.util.FORTRAN_integrate(t, x0_bk, f_f77, p0, rtol=tol, atol=tol)

# determine dominant frequencies for scaling time axis
print "Determining dominant frequencies in model behaviour..."
f_fw, psd_fw = nm.analysis.psd(t, x_arr_fw[:, 3*N_seg], timescale=1, detrend=True)
dom_freq_fw = np.abs(f_fw[np.argmax(psd_fw)])
f_bk, psd_bk = nm.analysis.psd(t, x_arr_bk[:, 3*N_seg], timescale=1, detrend=True)
dom_freq_bk = np.abs(f_bk[np.argmax(psd_bk)])

print "Plotting mechanical state during forward/backward waves..."
# plot forward and backward waves
plt.ioff()
fig = plt.figure("soliton solutions", figsize=(3.3, 7))
plt.clf()
plot = fig.add_subplot(211)
plt.cla()
plt.plot(t*dom_freq_fw, x_arr_fw[:, 3*N_seg:4*N_seg] + np.arange(N_seg), lw=2)
plt.plot(t*dom_freq_fw, x_arr_fw[:, 3*N_seg] + N_seg, lw=2)
plt.xlim(0, 5)
plt.ylim(-1, 12 + 5)
plt.yticks([0, 5, 10, 15])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.ylabel("axial displacement (segs)", fontsize=fontsize)

plot = fig.add_subplot(212)
plt.cla()
plt.plot(t*dom_freq_bk, x_arr_bk[:, 3*N_seg:4*N_seg] + np.arange(N_seg), lw=2)
plt.plot(t*dom_freq_bk, x_arr_bk[:, 3*N_seg] + N_seg, lw=2)
plt.ylabel("axial displacement (segs)", fontsize=fontsize)
plt.xlabel("time (s)", fontsize=fontsize)
plt.xlim(0, 5)
plt.ylim(-6, 12)
plt.yticks([-5, 0, 5, 10])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.tight_layout()
nm.util.hide_spines()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "soliton_solutions.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


"""
Plot neuromechanical state.
"""

print "Plotting neuromechanical state during forward/backward waves..."
stretch_arr_fw = np.diff(x_arr_fw[:, 3*N_seg:4*N_seg], axis=1)
stretch_rate_arr_fw = np.diff(x_arr_fw[:, 4*N_seg:5*N_seg], axis=1)

stretch_arr_bk = np.diff(x_arr_bk[:, 3*N_seg:4*N_seg], axis=1)
stretch_rate_arr_bk = np.diff(x_arr_bk[:, 4*N_seg:5*N_seg], axis=1)


plt.ioff()
plt.figure("forward neuromechanical state", figsize=(4, 6))
plt.clf()
plt.subplot(411)
plt.plot(t*dom_freq_fw, stretch_rate_arr_fw/3. + np.arange(10), c='k', lw=2)
plt.ylabel("stretch rate")
plt.xlim(0, 5)
plt.grid(False)

plt.subplot(412)
plt.plot(t*dom_freq_fw, x_arr_fw[:, :N_seg] + np.arange(11), c='k', lw=2)
plt.ylabel("SN")
plt.xlim(0, 5)
plt.grid(False)

plt.subplot(413)
plt.plot(t*dom_freq_fw, x_arr_fw[:, N_seg:2*N_seg] + np.arange(11), c='k', lw=2)
plt.ylabel("IN")
plt.xlim(0, 5)
plt.grid(False)

plt.subplot(414)
plt.plot(t*dom_freq_fw, x_arr_fw[:, 2*N_seg:3*N_seg] + np.arange(11), c='k', lw=2)
plt.ylabel("MN")
plt.xlim(0, 5)
plt.xlabel("time (s)")
plt.grid(False)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "forward_neuromech_state.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


plt.ioff()
plt.figure("backward neuromechanical state", figsize=(4, 6))
plt.clf()
plt.subplot(411)
plt.plot(t*dom_freq_bk, stretch_rate_arr_bk/3. + np.arange(10), c='k', lw=2)
plt.ylabel("stretch rate")
plt.xlim(0, 5)
plt.grid(False)

plt.subplot(412)
plt.plot(t*dom_freq_bk, x_arr_bk[:, :N_seg] + np.arange(11), c='k', lw=2)
plt.ylabel("SN")
plt.xlim(0, 5)
plt.grid(False)

plt.subplot(413)
plt.plot(t*dom_freq_bk, x_arr_bk[:, N_seg:2*N_seg] + np.arange(11), c='k', lw=2)
plt.ylabel("IN")
plt.xlim(0, 5)
plt.grid(False)

plt.subplot(414)
plt.plot(t*dom_freq_bk, x_arr_bk[:, 2*N_seg:3*N_seg] + np.arange(11), c='k', lw=2)
plt.ylabel("MN")
plt.xlim(0, 5)
plt.xlabel("time (s)")
plt.grid(False)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "backward_neuromech_state.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


"""
Bifurcation diagram for steady-state centre-of-mass momentum P_inf, with
feedback gain b as a control parameter.
"""

print "Running bifurcation analysis with feedback gain as control parameter..."

# set up ics
x0 = np.zeros(len(x0_fw))
n_ics = 10
ics = [2*(2*np.random.random(2*N_seg) - 1) for i in xrange(n_ics - 1)]
ics.append(np.zeros(2*N_seg))

DFT = (1./N_seg)*sp.fft(np.eye(N_seg))
md0_vec = np.real(DFT[0])
md1_vec = np.real(DFT[1])
md2_vec = np.imag(DFT[1])

output_fps = 30                                     # output sampling rate (Hz)
t_arr_scaled = t*dom_freq_fw                        # scaled time axis so that 1 wave = 1 second
samples_per_wave = np.searchsorted(t_arr_scaled, 1) # number of samples in a wave
decimation_step = samples_per_wave/output_fps       # step needed to achieve 30 fps

bs = np.linspace(-1, 10, 55)
P_infs = []
phi_infs = []

N_tot_sims = len(bs)*n_ics
n_sim = 1
for b in np.linspace(-1, 10, 55) :
    print ""
    for i in xrange(n_ics) :
        print "Running simulation for IC " + str(i + 1) + " of " + str(n_ics) +\
                " with reflex gain b=" + str(b) +\
                " (simulation " + str(n_sim) + " of " + str(N_tot_sims) + ")"
        # set reflex gain
        p0[2] = b

        # set initial conditions
        x0[3*N_seg:5*N_seg] = ics[i]

        # integrate
        x_arr = nm.util.FORTRAN_integrate(t, x0, f_f77, p0, rtol=tol, atol=tol)

        # calculate long-term average of centre of mass momentum
        P_inf = np.mean(x_arr[-3000:, 4*N_seg:5*N_seg])
        P_infs.append(P_inf)

        # project onto low frequency modes
        q1 = np.dot(x_arr[:, 3*N_seg:4*N_seg], md1_vec)
        q2 = np.dot(x_arr[:, 3*N_seg:4*N_seg], md2_vec)
        p1 = np.dot(x_arr[:, 4*N_seg:5*N_seg], md1_vec)
        p2 = np.dot(x_arr[:, 4*N_seg:5*N_seg], md2_vec)

        # calculate relative phase of low frequency modes
        q1_norm = q1/np.max(q1)
        q2_norm = q2/np.max(q2)
        p1_norm = p1/np.max(p1)
        p2_norm = p2/np.max(p2)
        phi1 = np.unwrap(np.arctan2(p1_norm, q1_norm))
        phi2 = np.unwrap(np.arctan2(p2_norm, q2_norm))
        phi_inf = (phi1 - phi2)[-1]
        phi_infs.append(phi_inf)

        # downsample output then save for future analysis
        x_arr_ds = x_arr[::decimation_step]
        del(x_arr, x_arr_ds)
        n_sim = n_sim + 1
P_infs = np.array(P_infs)
phi_infs = np.array(phi_infs)

print "Plotting bifurcation analysis results..."
# produce a plot showing centre of mass momentum AND modal phase difference
plt.ioff()
fig = plt.figure("symmetry breaking with phase", figsize=(3.5, 7))
plt.clf()
P_inf_thresh = 0.02
for b_i, P_infs_i, phi_infs_i in zip(bs, P_infs.reshape(-1, 10),
        phi_infs.reshape(-1, 10)) :
    for P_inf_i, phi_inf_i in zip(P_infs_i, phi_infs_i) :
        c = 'k'
        if P_inf_i > P_inf_thresh :
            c = 'royalblue'
        elif P_inf_i < -P_inf_thresh :
            c = 'red'
        plot1 = fig.add_subplot(211)
        plt.scatter(b_i, P_inf_i, c=c, edgecolor="none")
        plt.xlim(-1, 10)
        plt.yticks([-1, 0, 1])
        plt.ylabel("centre of mass momentum", fontsize=fontsize)
        plt.grid(False)
        plot2 = fig.add_subplot(212)
        plt.scatter(b_i, -(phi_inf_i % (2*np.pi) - np.pi), c=c, edgecolor="none")
        plt.xlim(-1, 10)
        plt.yticks([-np.pi, -np.pi/2., 0, np.pi/2., np.pi], 
          ["$-\pi$", "$-\\frac{\pi}{2}$", "0", "$\\frac{\pi}{2}$", "$\pi$"])
        plt.xlabel("reflex gain", fontsize=fontsize)
        plt.ylabel("modal phase difference", fontsize=fontsize)
        plt.grid(False)

plot1.tick_params(axis="both", which="major", labelsize=fontsize)
plot2.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "symmetry_breaking_phase.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()



"""
Visualise limit cycles using modal coordinates. 
"""

print "Visualising forward/backward limit cycles in modal coordinates..."
# set representative parameters
p0 = [-2, -2, 7, 1, 1, 2*np.pi, 0.01, 0.5]

DFT = (1./N_seg)*sp.fft(np.eye(N_seg))

md0_vec = np.real(DFT[0])
md1_vec = np.real(DFT[1])
md2_vec = np.imag(DFT[1])

x0[3*N_seg:4*N_seg] = np.zeros(N_seg)

plt.ioff()
fig = plt.figure("LC visualisation", figsize=(7, 7))
plt.clf()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 1)
ax.set_xticks([-1, 0, 1])
ax.set_ylim(-1, 1)
ax.set_yticks([-1, 0, 1])
ax.set_zlim(-1, 1)
ax.set_zticks([-1, 0, 1])
ax.set_xlabel("axial mode 1", fontsize=fontsize)
ax.set_ylabel("axial mode 2", fontsize=fontsize)
ax.set_zlabel("centre of mass momentum", fontsize=fontsize)
ax.tick_params(axis="both", which="major", labelsize=fontsize)

print "Numerically integrating from random initial conditions..."
n_plots = 40
for i in xrange(n_plots) :
    print str(i + 1) + "of" + str(n_plots)
    x0[4*N_seg:] = 10*2*(np.random.random(N_seg) - 0.5)
    x_arr = nm.util.FORTRAN_integrate(t, x0, f_f77, p0, rtol=tol, atol=tol)
    md0 = np.dot(md0_vec, x_arr[:, 4*N_seg:5*N_seg].T)
    md1 = np.dot(md1_vec, x_arr[:, 4*N_seg:5*N_seg].T)
    md2 = np.dot(md2_vec, x_arr[:, 4*N_seg:5*N_seg].T)
    plt.figure("LC visualisation")
    c = 'k'
    if md0[-1] > 0 :    # if trajectory fell onto forward LC, plot in blue
        c = 'royalblue'
    elif md0[-1] < 0 :  # if trajectory fell onto backward LC, plot in red
        c = 'red'
    plt.plot(md1[::decimation_step], 
             md2[::decimation_step], 
             md0[::decimation_step], c=c, lw=0.1)

print "Plotting..."
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "LC_modal_visualisation.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()

