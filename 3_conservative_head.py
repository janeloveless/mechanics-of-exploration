#! /usr/bin/env python2

import time
import itertools as it
import numpy as np
import scipy as sp
import sympy as sy
from sympy import S
import sympy.physics.mechanics as mech
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import neuromech as nm


"""
Simulate and analyse the motion of the reduced mechanical model of the head over
a range of amplitudes, under the assumption of energy conservation (no friction
or driving forces).
"""

"""
Set some parameters. 

Some others will be defined later in the code, where it is more convenient.
"""

print "Setting parameters..."

# mechanical parameters
E0 = 0.5                    # total mechanical energy
lam = np.exp(1)/6           # ratio of transverse to axial natural frequencies

# filesystem parameters
PLOT_PATH = "./data/output/3_conservative_head/"
F_PATH = "./FORTRAN_sources/"

# plotting parameters
fontsize = 12
output_dpi = 450
SAVE_PLOTS = True
SHOW_PLOTS = False


"""
Set up model, then extract dynamics, state variables, parameters, etc.
"""

print "Constructing model of head motion..."

model = nm.model.ConservativeHead()
    
f = model.f                                 # dynamics
x = q, phi, p_q, p_phi = model.x            # state variables
H = model.H                                 # Hamiltonian
params = model.parameters                   # model parameters


"""
Derive equations of motion, compile to FORTRAN, then compile to binary.
"""

print "Compiling RHS function to intermediate FORTRAN source code..."
f_src = model.FORTRAN_f(verbose=True)

# save FORTRAN source code for future usage
with open(F_PATH + "3_conservative_head.f", "w") as src_file :
    src_file.write(f_src)

f_src = open(F_PATH + "3_conservative_head.f").read()

print "Compiling RHS function FORTRAN source code to binary..."
f_f77 = nm.util.FORTRAN_compile(f_src)


"""
Set simulation parameters.
"""

tol = 10**-8                            # absolute and relative tolerance for numerical integrator
t_arr = np.linspace(0, 16000, 32*40000) # simulation time axis

p0 = [1, lam]                  # set parameters epsilon, lambda


"""
Find initial conditions which are consistent with a given value of energy.
"""

E = sy.symbols("E")                         # symbol for total mechanical energy
E_shell = sy.Eq(E, H)                       # implicit equation for the model's energy shell

def x0_gen(q0, phi0, p_phi0=0, E0=E0, params=params, p0=p0) :
    """
    This uses the implicit equation E = H(q, phi, p_q, p_phi) to find the
    initial value of p_q needed to obtain a given value of the total energy,
    given initial values of q, phi, and p_phi.

    Note that this problem does not always have a solution, so the output should
    be checked.
    """
    # energy shell for given total energy, (partial) initial conditions, and parameters
    E_shell_subs = E_shell.subs(zip(params, p0)).subs(E, E0)
    E_shell_subs = E_shell_subs.subs(q, q0).subs(phi, phi0).subs(p_phi, p_phi0)
    # solve for initial value of p_q
    p_q0 = sy.solve(E_shell_subs, p_q)[0]
    return [q0, phi0, p_q0, p_phi0]

x0_repr = x0_gen(-0.4, -0.4)


"""
Produce Poincare sections for several values of epsilon.
"""

plt.ioff()
N_ICs = 60                                          # number of initial conditions (should be even!)
eps_num_list = [0.0, 0.33, 0.67, 1.0]               # values of epsilon to use
N_tot_sections = (N_ICs + 12)*len(eps_num_list)     # total number of trajectories per plot
n_section = 1                                       # number of trajectories currently plotted
for eps_num in eps_num_list :
    p0[0] = eps_num
    print "producing plot for e =", eps_num

    # find the energy shell corresponding to our choice of parameters, over the
    # configuration space, then find major and minor axes of the shell
    E_shell_conf = E_shell.subs(zip(params, p0)).subs(E, E0).subs(p_phi, 0).subs(p_q, 0)
    q_max = float(np.max(sy.solve(E_shell_conf.subs(phi, 0), q)))
    phi_max = float(np.max(sy.solve(E_shell_conf.subs(q, 0), phi)))
    
    # produce initial conditions which are randomly distributed on the energy shell
    rhos = np.random.random(N_ICs/2)
    thetas = 2*np.pi*np.random.random(N_ICs/2)
    q0s = np.sqrt(rhos)*np.cos(thetas)
    q0s = q_max*np.concatenate([q0s, q0s])
    phi0s = phi_max*np.sqrt(rhos)*np.sin(thetas)
    phi0s = np.concatenate([phi0s, -phi0s])

    # add some special q0s, phi0s
    q0s = list(q0s) + [0, 0.2, 0.2, -0.2, -0.2, -0.6, -0.6, -0.6, -0.6, 0.6,
            0.6] + [x0_repr[0]]
    phi0s = list(phi0s) + [0, -0.6, 0.6, -0.45, 0.45, -0.42, 0.42, -0.45, 0.45,
            -0.45, 0.45] + [x0_repr[1]]

    # find and use an expression for p_q when on-shell, with given configuration variables q, phi
    p_q_expr = sy.solve(E_shell.subs(zip(params, p0)).subs(E, E0).subs(p_phi, 0), p_q)[0]
    p_q_lam = sy.lambdify([q, phi], p_q_expr)
    p_q0s = [p_q_lam(q0, phi0) for q0, phi0 in zip(q0s, phi0s)]
    x0s = np.array([q0s, phi0s, p_q0s, np.zeros(len(q0s))]).T
    
    cs = np.random.rand(len(x0s), 3, 1)
    
    # find the energy shell in the configuration plane when momenta are set equal
    # to zero -- this is useful for visualisation of poincare plots
    E_shell_planar = E_shell.subs(E, E0).subs(zip(params, p0)).subs(p_q, 0).subs(p_phi, 0)
    E_shell_planar_explicit = sy.solve(E_shell_planar, phi)[0]
    E_shell_lims = sy.solve(E_shell_planar_explicit, q)
    qs = np.linspace(float(E_shell_lims[0])*0.999999,
                     float(E_shell_lims[1])*0.999999, 1000)
    
    E_shell_planar_lam = sy.lambdify(q, E_shell_planar_explicit)
    E_shell_planar_num = np.vectorize(E_shell_planar_lam)
    E_shell_planar_arr = np.concatenate([E_shell_planar_num(qs), -E_shell_planar_num(qs[::-1])])
    E_shell_planar_qs = np.concatenate([qs, qs[::-1]])
    
    # define function for producing poincare plot with given parameters
    def poincare_plot(x_arr, epsilon=0.001, E_shell_planar_qs=E_shell_planar_qs, 
                      E_shell_planar_arr=E_shell_planar_arr, s=10, c='k',
                      E_c='blue', E0=E0, params=params, p0=p0) :
        p_phi_zero_crossings = np.abs(x_arr[:, 3]) < epsilon
        plt.scatter(x_arr[:, 1][p_phi_zero_crossings],
                    x_arr[:, 0][p_phi_zero_crossings], s=s, c=c, edgecolors='none')
        plt.tight_layout()
    
    # generate poincare plots!
    fig = plt.figure("Poincare e=" + str(p0[0]), figsize=(5, 5))
    plot = fig.add_subplot(111)
    plot.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.plot(E_shell_planar_arr, E_shell_planar_qs, c='grey', lw=0.5)
    plt.text(np.pi/2., 0.85, "$\epsilon = " + str(p0[0]) + "$")
    plt.xlim(-1.05*phi_max, 1.05*phi_max)
    plt.xticks([-np.pi/2., 0, np.pi/2.], ["$-\pi/2$", "$0$", "$\pi/2$"])
    plt.xlabel("$\phi$ (head bend, rad)", fontsize=fontsize)
    plt.ylabel("$q$ (head strain, dimensionless)", fontsize=fontsize)
    plt.ylim(-1.05*q_max, 1.05*q_max)
    plt.yticks([-1, 0, 1])
    plt.grid(False)
    for i in xrange(len(x0s)) :
        print "Poincare section for IC " + str(i + 1) + " of " + str(len(x0s)) +\
                ", e = " + str(p0[0]) +\
                " (section " + str(n_section) + " of " + str(N_tot_sections) + ")"
        x0 = x0s[i]
        c = cs[i].T[0]
        x_arr = nm.util.FORTRAN_integrate(t_arr, x0, f_f77, p0, atol=tol, rtol=tol)
        poincare_plot(x_arr, s=0.5, epsilon=0.0005, c=c)
        n_section = n_section + 1

    plt.scatter(x0_repr[1], x0_repr[0], s=30, c='k', edgecolors="none")
    plt.tight_layout()
    nm.util.hide_spines()
    if SAVE_PLOTS : plt.savefig(PLOT_PATH + "poincare_e" + str(p0[0]) + ".png", dpi=400)
    if not SHOW_PLOTS : plt.close()

plt.ion()
plt.show()

    
"""
Perform analysis of chaotic behaviour using Lyapunov exponent, power spectrum,
and autocorrelation.
"""

print "Analysing chaotic behaviour using Lyapunov exponent, power spectrum, and autocorrelation..."

c="k"
tol = 1*10**-8
eps_num_list = [0.0, 0.33, 0.67, 1.0]

for eps in eps_num_list : 
    print "Setting epsilon = " + str(eps) + "..."
    p0[0] = eps
    
    print "Running Lyapunov exponent estimation algorithm..."
    lce_analysis = nm.analysis.lce_estimate(x0, f_f77, p0, t_step=(t_arr[1] -
        t_arr[0])/4., pb_step=200, n_pb=2000, n_ic_steps=100, tol=tol, debug=False,
        n_pb_discard=100, d0=10**-7)
        
    print "Scaling Lyapunov exponent estimates..."
    T = 2*np.pi
    lce_bpw = T*lce_analysis[0][0]/np.log(2)
    lt_lce_estimates_bpw = T*lce_analysis[1]/np.log(2)
    ft_lce_estimates_bpw = T*lce_analysis[2]/np.log(2)
         
    print "Generating representative trajectory..."
    x_arr = nm.util.FORTRAN_integrate(t_arr, x0, f_f77, p0, atol=tol, rtol=tol)
        
    print "Calculating power spectra..."
    psd_q = nm.analysis.psd(t_arr, x_arr[:, 0], timescale=1)
    psd_phi = nm.analysis.psd(t_arr, x_arr[:, 1], timescale=1)

    print "Calculating autocorrelation..."
    corr_q = nm.analysis.correlation(t_arr, x_arr[:, 0], x_arr[:, 0])
    corr_phi = nm.analysis.correlation(t_arr, x_arr[:, 1], x_arr[:, 1])
        
    print "Plotting..."
    plt.ioff()
    fig = plt.figure("chaos analysis e=" + str(p0[0]), figsize=(5, 5))
    plt.clf()
    plt.subplot(321)
    plt.cla()
    plt.plot(lt_lce_estimates_bpw, c=c, lw=1)
    plt.axhline(lce_bpw, c="b", lw=2, alpha=0.5)
    plt.xlim(0, 2000)
    plt.xlabel("iteration")
    plt.ylim(-0.5, 2)
    plt.yticks([0, 1, 2])
    plt.text(200, 1.5, "MLCE = " + str(np.round(lce_bpw, 2)) + " bits s$^{-1}$")
    plt.ylabel("MLCE (bits s$^{-1}$)")
    plt.grid(False)
    
    #plt.subplot(322)
        
    plt.subplot(323)
    plt.plot(psd_q[0]*T, np.log(psd_q[1]), c=c, lw=0.5)
    plt.xlim(0, 3)
    plt.ylim(-10, 20)
    plt.yticks([-10, 0, 10, 20])
    plt.xticks([0, 1, 2, 3])
    #plt.xlabel("frequency (Hz)")
    plt.ylabel("log PSD $q$")
    plt.grid(False)
    
    plt.subplot(325)
    plt.plot(psd_phi[0]*T, np.log(psd_phi[1]), c=c, lw=0.5)
    plt.xlim(0, 2)
    plt.ylim(-10, 20)
    plt.yticks([-10, 0, 10, 20])
    plt.xticks([0, 1, 2, 3])
    plt.xlabel("frequency (Hz)")
    plt.ylabel("log PSD $\phi$")
    plt.grid(False)
        
    plt.subplot(324)
    plt.plot(corr_q[0]/T, corr_q[1]/np.max(corr_q[1]), c=c, lw=0.5)
    plt.yticks([-1, 0, 1])
    #plt.xlabel("time lag (s)")
    plt.ylabel("autocorr. $q$")
    plt.xlim(-1000, 1000)
    plt.xticks([-1000, -500, 0, 500, 1000])
    plt.grid(False)
    
    plt.subplot(326)
    plt.plot(corr_phi[0]/T, corr_phi[1]/np.max(corr_phi[1]), c=c, lw=0.5)
    plt.yticks([-1, 0, 1])
    plt.xlabel("time lag (s)")
    plt.ylabel("autocorr. $\phi$")
    plt.xlim(-1000, 1000)
    plt.xticks([-1000, -500, 0, 500, 1000])
    plt.grid(False)
    
    nm.util.hide_spines()
    plt.tight_layout()
    
    if SAVE_PLOTS : plt.savefig(PLOT_PATH + "analysis_e" + str(p0[0]) + ".png", dpi=output_dpi)
    if not SHOW_PLOTS : plt.close()
    plt.ion()
    plt.show()
    time.sleep(0.5)

