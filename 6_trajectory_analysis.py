#! /usr/bin/env python2

import os
import random
import numpy as np
import scipy as sp
import scipy.linalg
import sympy as sy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import powerlaw as pl
import neuromech as nm

"""
Perform analysis on output of 5_exploration.py
"""

"""
Useful parameters, definitions, constants.
"""

# filesystem parameters
DATA_PATH = "./data/output/5_exploration/simulation_outputs/"
LCE_ANALYSIS_PATH = "./data/output/5_exploration/LCE_analysis/"
PLOT_PATH = "./data/output/6_trajectory_analysis/"
data_files = os.listdir(DATA_PATH)      # list of simulation output filenames

# mechanical parameters
N = 12                                  # number of masses
N_DOF = 2*N                             # number of mechanical degrees of freedom
N_seg = N - 1                           # number of segments
N_neuron = 3*N_seg                      # number of neurons

# analysis parameters
n = len(data_files)                     # number of simulation results to load/analyse
D_c_N_samples = 1500                    # number of samples to use for
                                        # correlation dimension estimate
turn_threshold = 20                     # body bend threshold for turning (degrees)
minimum_run_length = 0.5                # minimum run length (seconds)

# useful indices for data files
tx_i = N_neuron + 0                     # index of tail x coordinate
ty_i = N_neuron + 1                     # index of tail y coordinate
mx_i = N_neuron + 12                    # index of midpoint x coordinate
my_i = N_neuron + 13                    # index of midpoint y coordinate
hx_i = N_neuron + 22                    # index of head x coordinate
hy_i = N_neuron + 23                    # index of head y coordinate
t_px_i = N_neuron + 24                  # index of tail momentum, x component
t_py_i = N_neuron + 25                  # index of tail momentum, y component
h_px_i = N_neuron + 46                  # index of head momentum, x component
h_py_i = N_neuron + 47                  # index of head momentum, y component

# plotting parameters
SAVE_PLOTS = True
SHOW_PLOTS = True
fontsize = 12
output_dpi = 450


"""
Perform analysis.
"""

## quantities to keep track of :
COMs = []                                       # centre of mass
tail_speeds = []                                # tail speed
body_bends = []                                 # body bend
bending_velocities = []                         # bending angular velocity
Cs = []                                         # instantaneous curvature
As = []                                         # angular speed
run_lengths = []                                # time interval between successive turns
tail_speed_psds = []                            # tail speed power spectral density
f_tail_speed_psds = []                          # frequency axes
bending_velocities_psds = []                    # bending angular velocity power spectral density
f_bending_velocities_psds = []                  # frequency axes
head_seg_corrs = []                             # head segment bend autocorrelation
t_lags_head_seg_corrs = []                      # autocorrelation time lag axes
head_seg_psds = []                              # head segment bend power spectral density
f_head_seg_psds = []                            # frequency axes
full_state_D_c = []                             # full state correlation dimension
tail_seg_corrs = []                             # tail segment strain autocorrelation
t_lags_tail_seg_corrs = []                      # autocorrelation time lag axes
tail_seg_psds = []                              # tail segment strain power spectral density
f_tail_seg_psds = []                            # frequency axes
COM_box_counting_dimensions = []                # box-counting dimension of COM trajectory
SDs = []                                        # squared displacements
Ls = []                                         # path lengths
Ts = []                                         # tortuosity indices


for i in xrange(n) :
    print "analysing trajectory " + str(i + 1) + " of " + str(n) + "..."
    sr, x = np.load(DATA_PATH + data_files[i])      # load sampling rate and state trajectory
    sr = sr[0]                                      # unpack sampling rate
    t = (1./sr)*np.arange(0, len(x), 1)             # construct time axis

    # form some convenient views on the state trajectory
    n_x = x[:, :N_neuron]                           # slice out neural state
    SN_x = x[:, :N_seg]                             # slice out sensory neuron state
    IN_x = x[:, N_seg:2*N_seg]                      # slice out interneuron state
    MN_x = x[:, 2*N_seg:3*N_seg]                    # slice out motor neuron state
    m_x = x[:, N_neuron:(N_neuron + 2*N_DOF)]       # slice out mechanical state
    q = m_x[:, :N_DOF]                              # slice out mechanical configuration
    q_vec = q.reshape(len(t), -1, 2)                # configuration as [x, y] vectors
    

    ## centre of mass (path) analysis
    print "    path analysis..."
    # compute centre of mass trajectory
    COM_x = np.mean(q[:, 0::2], axis=1)
    COM_y = np.mean(q[:, 1::2], axis=1)
    COM = np.array([COM_x, COM_y]).T
    COMs.append(COM)
    # compute instantaneous curvature and angular speed
    t_new, C, A = nm.analysis.curvature_angular_speed_analysis(t, COM,
            discard=1500)
    Cs.append(C)
    As.append(A)
    # box-counting dimension of path
    D_b = nm.analysis.box_counting_dimension_estimate_2d(COM, 
            min_scale=3, num_scales=100)
    COM_box_counting_dimensions.append(D_b)
    # diffusion distance calculations
    D = np.linalg.norm(COM - COM[0], axis=1)    # net displacement
    SD = D**2                                   # squared displacement
    SDs.append(SD)
    # path distance
    L = np.sum(np.linalg.norm(np.diff(COM, axis=0), axis=1))
    Ls.append(L)
    # tortuosity index of path
    T = 1 - D[-1]/L
    Ts.append(T)
    # NOTE sinuousity index of path can only be computed later, once we have
    # an estimate for the MSD from the whole population


    ## 2-segment analysis
    print "    2-segment analysis..."
    # compute tail speed, body bend, and angular velocity
    ts = nm.analysis.tail_speed(x, t_px_i=t_px_i, t_py_i=t_py_i)
    tail_speeds.append(ts)
    bb = nm.analysis.body_bend(x, tx_i=tx_i, ty_i=ty_i, mx_i=mx_i, my_i=my_i,
                                  hx_i=hx_i, hy_i=hy_i)
    body_bends.append(bb)
    av = nm.analysis.head_angular_velocity(x, mx_i=mx_i, my_i=my_i, hx_i=hx_i,
                                hy_i=hy_i, h_px_i=h_px_i, h_py_i=h_py_i)
    bending_velocities.append(av)
    # compute run lengths
    rl = nm.analysis.run_lengths(t, bb, threshold=turn_threshold,
            minimum_length=minimum_run_length)
    run_lengths.append(rl)
    # compute power spectral densities
    f_ts, S_ts = nm.analysis.psd(t, ts, timescale=1)
    f_av, S_av = nm.analysis.psd(t, av, timescale=1)
    tail_speed_psds.append(S_ts)
    f_tail_speed_psds.append(f_ts)
    bending_velocities_psds.append(S_av)
    f_bending_velocities_psds.append(f_av)


    ## internal chaos analysis
    print "    chaos analysis..."
    # compute segment lengths and bending angles
    lengths = np.linalg.norm(np.diff(q_vec, axis=1), axis=2)
    angles = np.diff(np.arctan2(np.diff(q_vec, axis=1)[:, :, 0],
                                np.diff(q_vec, axis=1)[:, :, 1]), axis=1)
    angles = np.unwrap(angles, axis=0)
    # head bend power spectral density
    f_hb, S_hb = nm.analysis.psd(t, angles[:, -1], timescale=1)
    head_seg_psds.append(S_hb)
    f_head_seg_psds.append(f_hb)
    # head bend autocorrelation
    t_lag_hb, R_hb = nm.analysis.correlation(t, angles[:, -1], angles[:, -1])
    head_seg_corrs.append(R_hb)
    t_lags_head_seg_corrs.append(t_lag_hb)
    # tail strain power spectral density
    f_tl, S_tl = nm.analysis.psd(t, lengths[:, 0], timescale=1)
    tail_seg_psds.append(S_tl)    
    f_tail_seg_psds.append(f_tl)
    # tail strain autocorrelation
    tl_mean = np.mean(lengths[:, 0])
    t_lag_tl, R_tl = nm.analysis.correlation(t, lengths[:, 0] - tl_mean, 
                                                lengths[:, 0] - tl_mean)
    tail_seg_corrs.append(R_tl)
    t_lags_tail_seg_corrs.append(t_lag_tl)
    # correlation dimension
    try :
        c_x = np.copy(m_x)
        c_x[:, 0:N_DOF:2] = (c_x[:, 0:N_DOF:2].T - COM_x).T
        c_x[:, 1:N_DOF:2] = (c_x[:, 1:N_DOF:2].T - COM_y).T
        c_x_sample = np.array(random.sample(c_x, D_c_N_samples))
        D_c = nm.analysis.correlation_dimension_estimate(c_x_sample)
        full_state_D_c.append(D_c)
    except :
        print "    WARNING couldn't estimate attractor dimension!"

    print "    current attractor dimension estimate = " + str(np.mean(full_state_D_c))
    print "    current path dimension estimate = " + str(np.mean(COM_box_counting_dimensions))


"""
Further analysis.
"""

# convert to numpy arrays
COMs = np.array(COMs)
Cs = np.array(Cs)
As = np.array(As)
Ts = np.array(Ts)
tail_speeds = np.array(tail_speeds)
bending_velocities = np.array(bending_velocities)
body_bends = np.array(body_bends)

tail_speed_psds = np.array(tail_speed_psds)
bending_velocities_psds = np.array(bending_velocities_psds)

full_state_D_c = np.array(full_state_D_c)

tail_seg_corrs = np.array(tail_seg_corrs)
t_lags_tail_seg_corrs = np.array(t_lags_tail_seg_corrs)
head_seg_corrs = np.array(head_seg_corrs)
t_lags_head_seg_corrs = np.array(t_lags_head_seg_corrs)

tail_seg_psds = np.array(tail_seg_psds)
f_tail_seg_psds = np.array(f_tail_seg_psds)
head_seg_psds = np.array(head_seg_psds)
f_head_seg_psds = np.array(f_head_seg_psds)

# concatenate output arrays
ts_concat = np.concatenate(tail_speeds)
av_concat = np.concatenate(bending_velocities)
bb_concat = np.concatenate(body_bends)
bb_concat2 = np.concatenate(body_bends[:, 500:])    # body bends, discarding correlated transient
rl_concat = np.concatenate(run_lengths)
t_concat = np.arange(0, len(ts_concat), 1)*(1./sr)

# compute 2-segment histograms
ts_min = np.min(ts_concat); ts_max = np.max(ts_concat)
av_min = np.min(av_concat); av_max = np.max(av_concat)
bb_min = np.min(bb_concat); bb_max = np.max(bb_concat)
bb_min2 = np.min(bb_concat2); bb_max2 = np.max(bb_concat2)
rl_min = np.min(rl_concat); rl_max = np.max(rl_concat)

ts_bins = np.linspace(ts_min, ts_max, 100)
av_bins = np.linspace(av_min, av_max, 100)
bb_bins = np.linspace(bb_min, bb_max, 100)
bb_lim2 = max(np.abs(bb_min2), np.abs(bb_max2))
bb_bins2 = np.linspace(-bb_lim2, bb_lim2, 150)
rl_bins = np.linspace(rl_min, rl_max, 20)

ts_pdf, bin_edges = np.histogram(ts_concat, bins=ts_bins, density=True)
av_pdf, bin_edges = np.histogram(av_concat, bins=av_bins, density=True)
bb_pdf, bin_edges = np.histogram(bb_concat, bins=bb_bins, density=True)
bb_pdf2, bin_edges = np.histogram(bb_concat2, bins=bb_bins2, density=True)
rl_pdf, bin_edges = np.histogram(rl_concat, bins=rl_bins, density=True)

ts_bins = ts_bins[:-1] + (ts_bins[1] - ts_bins[0])/2.
av_bins = av_bins[:-1] + (av_bins[1] - av_bins[0])/2.
bb_bins = bb_bins[:-1] + (bb_bins[1] - bb_bins[0])/2.
bb_bins2 = bb_bins2[:-1] + (bb_bins2[1] - bb_bins2[0])/2.
rl_bins = rl_bins[:-1] + (rl_bins[1] - rl_bins[0])/2.

# compute exponential fit to run length distribution
rl_fit = sp.stats.linregress(rl_bins[:14], np.log(rl_pdf[:14]))
rl_fit_slope = rl_fit[0]
rl_fit_intercept = rl_fit[1]

# evaluate fit of run-length distribution by exponential, power law, etc.
rl_fit_analysis = pl.Fit(rl_concat, xmin=1, xmax=1000)
rl_exp_vs_pl = rl_fit_analysis.distribution_compare("exponential", "power_law")
rl_exp_vs_st_exp = rl_fit_analysis.distribution_compare("exponential", "stretched_exponential")
rl_exp_anderson = sp.stats.anderson(rl_concat, dist="expon")

# evaluate fit of body-bend distribution by Gaussian, von Mises, etc.
bb_mean = np.mean(bb_concat2)
bb_var = np.std(bb_concat2)**2
bb_skew = sp.stats.skew(bb_concat2)
bb_kurtosis = sp.stats.kurtosis(bb_concat2)
bb_norm_anderson = sp.stats.anderson(bb_concat, dist="norm")

bb_vm_fit = sp.stats.vonmises.fit(bb_concat2[::100], 100, floc=0, fscale=1)
bb_vm_dist = sp.stats.vonmises(bb_vm_fit[0], loc=0, scale=1)
bb_wrapcauchy_fit = sp.stats.wrapcauchy.fit((bb_concat2[::100]) % (2*np.pi), 0.81, fscale=1, floc=0)
bb_wrapcauchy_dist = sp.stats.wrapcauchy(bb_wrapcauchy_fit[0], loc=0, scale=bb_wrapcauchy_fit[2])
bb_cauchy_fit = sp.stats.cauchy.fit(bb_concat2[::100], floc=0)
bb_cauchy_dist = sp.stats.cauchy(loc=bb_cauchy_fit[0], scale=bb_cauchy_fit[1])

bb_vm_logL = -sp.stats.vonmises.nnlf(bb_vm_fit, bb_concat2)
bb_wrapcauchy_logL = -sp.stats.wrapcauchy.nnlf(bb_wrapcauchy_fit, bb_concat2 % (2*np.pi))

# compute exponent and r^2 for putative curvature--angular speed power law
# do this for low curvature and all curvatures separately
logC_cutoff = -3
C = np.concatenate(Cs)
A = np.concatenate(As)
C_axis = np.linspace(np.min(Cs), np.max(Cs), 5000)
CA_fit_high_C = sp.stats.linregress(np.log10(C),
                                    np.log10(A))
CA_fit_high_C_slope = CA_fit_high_C[0]
CA_fit_high_C_intercept = CA_fit_high_C[1]
CA_fit_high_C_r2 = CA_fit_high_C[2]

CA_fit_low_C = sp.stats.linregress(np.log10(C[np.log10(C) < logC_cutoff]),
                                    np.log10(A[np.log10(C) < logC_cutoff]))
CA_fit_low_C_slope = CA_fit_low_C[0]
CA_fit_low_C_intercept = CA_fit_low_C[1]
CA_fit_low_C_r2 = CA_fit_low_C[2]

# compute mean-square displacement and diffusion coefficient
SDs = np.array(SDs)
MSD = np.mean(SDs, axis=0)
MSD_fit = sp.stats.linregress(t[len(MSD)/2:], MSD[len(MSD)/2:])
MSD_fit_slope = MSD_fit[0]
MSD_fit_intercept = MSD_fit[1]
diffusion_coefficient = MSD_fit_slope/4.

# compute MSD after discarding initial acceleration to evaluate anomalous
# diffusion (the initial acceleration gives the full MSD a >2 exponent)
def MSD_i(t, COMs, i=0) :
    SDs_i = []
    for COM in COMs :
        D = np.linalg.norm(COM[i:] - COM[i], axis=1)  # net displacement
        SD = D**2                                     # squared displacement
        SDs_i.append(SD)
    MSD_i = np.mean(SDs_i, axis=0)
    t_i = t[i:] - t[i]
    return t_i, MSD_i

t_anomdiff, MSD_anomdiff = MSD_i(t, COMs, len(MSD)/3)


# compute mean power spectral densities
mean_tail_speed_psd = np.mean(tail_speed_psds, axis=0)
std_tail_speed_psd = np.std(tail_speed_psds, axis=0)
f_mean_tail_speed_psd = f_tail_speed_psds[0]
mean_bending_velocities_psd = np.mean(bending_velocities_psds, axis=0)
std_bending_velocities_psd = np.std(bending_velocities_psds, axis=0)
f_mean_bending_velocities_psd = f_bending_velocities_psds[0]

# compute correlation dimension distribution
full_state_D_c_bins = np.linspace(1, 4, 120)
full_state_D_c_bin_width = full_state_D_c_bins[1] - full_state_D_c_bins[0]
full_state_D_c_pdf = np.histogram(full_state_D_c, bins=full_state_D_c_bins,
        normed=True)
full_state_D_c_mean = np.mean(full_state_D_c)
full_state_D_c_median = np.median(full_state_D_c)
full_state_D_c_mode = full_state_D_c_pdf[1][np.argmax(full_state_D_c_pdf[0])] + full_state_D_c_bin_width/2.

"""
Print analysis results.
"""

print "diffusion coefficient :", diffusion_coefficient
print "path dimension (mean, var) :", (np.mean(COM_box_counting_dimensions),
                                         np.std(COM_box_counting_dimensions)**2)
print "path tortuosity (mean, var) :", (np.mean(Ts), np.std(Ts)**2)

print "attractor dimension (mean, median, mode) :", np.mean(full_state_D_c),
np.median(full_state_D_c),
full_state_D_c_pdf[1][np.argmax(full_state_D_c_pdf[0])] + full_state_D_c

print "run length exponential vs. power law :", rl_exp_vs_pl
print "run length exponential lambda :", rl_fit_slope
print "run length exponential test (Anderson) p-value :", rl_exp_anderson

print "body bend mean :", bb_mean
print "body bend var :", bb_var
print "body bend skew :", bb_skew
print "body bend kurtosis :", bb_kurtosis
print "body bend normality test (Anderson) p-value :", rl_exp_anderson


"""
Plot analysis results.
"""

################################################################################################

n_plot = 13
plt.ioff()
fig = plt.figure("initial conditions", figsize=(2.2, 2.2))
plt.clf()
plot = fig.add_subplot(111)
for i in xrange(n_plot) :
    print "plotting initial conditions " + str(i + 1) + " of " + str(n_plot) + "..."
    sr, x = np.load(DATA_PATH + data_files[i])      # load sampling rate and state trajectory
    plt.plot(x[0][N_neuron:N_neuron + N_DOF:2], x[0][N_neuron + 1:N_neuron +
        N_DOF:2], c='k', lw=1, alpha=0.2, marker="o", markersize=3)
plt.xlim(-1, 13)
plt.ylim(-7, 7)
plt.xlabel("x (segment lengths)")
plt.ylabel("y (segment lengths)")
plt.xticks([0, 4, 8, 12])
plt.yticks([-4, 0, 4])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "initial_conditions.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


n_plot = 1000
plt.ioff()
fig = plt.figure("centre of mass trajectories", figsize=(8, 8))
plt.clf()
plot = fig.add_subplot(111)
#for COM in COMs[:n_plot] :
#    plt.plot(COM[:, 0], COM[:, 1], lw=0.2, c='k', zorder=2, alpha=0.5)
for COM in COMs[:n_plot] :
    plt.scatter(COM[:, 0][-1], COM[:, 1][-1], s=25, c='LightGray', zorder=1,
            edgecolors="none")

COM_reps = [COMs[10],
            COMs[100],
#            COMs[201],
            COMs[202],
            COMs[203],
            COMs[204],
            COMs[210]] # end close to start

colors = ["Gray", "Red", 
            "RoyalBlue", 
            "Orange", 
#          "LimeGreen", 
          "ForestGreen", "MediumTurquoise", "Navy", "Gray", "DimGray", "Black"]

for COM, c in zip(COM_reps, colors) : 
    plt.scatter(COM[0][0], COM[0][1], s=50, c='k', zorder=4,
            edgecolors="none")
    plt.plot(COM[:, 0], COM[:, 1], lw=3, zorder=3, c=c)
    plt.scatter(COM[-1][0], COM[-1][1], s=50, c=c, zorder=4,
            edgecolors="none")

plt.xlabel("x (segment lengths)", fontsize=fontsize)
plt.ylabel("y (segment lengths)", fontsize=fontsize)
plt.xlim(-400, 600)
plt.ylim(-500, 500)
plt.grid(False)
plt.text(380, 450, "$n = 1000$ paths")
plt.text(380, 420, "$t = 2.5$ minutes")
plt.text(380, 390, "$\\delta_0 \\leq 1\\times 10^{-8} segs$")
#plt.text(-150, -10, "\\textbf{START}")
plt.scatter(-365, 450 + 10, s=50, c='k', zorder=4, edgecolors="none")
plt.text(-335, 450, "starting position")

plt.scatter(-365, 420 + 10, s=25, c='LightGray', zorder=4, edgecolors="none")
plt.text(-335, 420, "final position")

for i in xrange(len(COM_reps)) :
    plt.scatter(-380 + i*5, 390 + 10, c=colors[i], s=50, edgecolors="none")
plt.text(-335, 390, "representative paths")

plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "centre_of_mass_trajectories_rep.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()

################################################################################################

plt.ioff()
fig = plt.figure("search analysis", figsize=(3.3, 8))
plt.clf()
plot = fig.add_subplot(411)
plt.cla()
plt.scatter(COM_box_counting_dimensions, Ts, c='k', s=5, alpha=0.3,
        edgecolors="none")
plt.xlabel("fractal dimension", fontsize=fontsize)
plt.ylabel("tortuosity", fontsize=fontsize)
plt.xticks([1, 1.5, 2])
plt.xlim(1, 2)
plt.yticks([0, 0.5, 1])
plt.ylim(0, 1)
plt.axvline(np.mean(COM_box_counting_dimensions), c='r', lw=2, alpha=0.1)
plt.axhline(np.mean(Ts), c='b', lw=2, alpha=0.2)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()

plot = fig.add_subplot(412)
plt.plot(t, SDs.T/10000., c='gray', lw=0.01)
plt.plot(t, MSD/10000., c='k', lw=6)
plt.plot(t, (MSD_fit_slope*t + MSD_fit_intercept)/10000., lw=2, c='r')
plt.plot(t, 0.55*10**-3*t**2, lw=2, c='royalblue')
plt.ylim(0, 10)
plt.xlabel("time (s)", fontsize=fontsize)
plt.ylabel("$\\langle d^2 \\rangle$ (segs$^{2}\\times 10^4$)", fontsize=fontsize)
plt.xlim(0, t[:len(MSD)][-1])
#plt.text(10, 9, "$D \\approx 144$ segs$^2 s^{-1}$")
plt.grid(False)
plt.xticks([0, 50, 100, 150])
plt.yticks([0, 5, 10])
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(414)
plt.cla()
plt.plot(rl_bins[:13], np.log(rl_pdf[:13]), lw=4, c='k')
plt.plot(rl_bins[:13], (rl_fit_slope*rl_bins[:13] + rl_fit_intercept), lw=2, c='r')
plt.xlim(0, 100)
plt.xlabel("run length (s)", fontsize=fontsize)
plt.ylabel("log prob. dens.", fontsize=fontsize)
#plt.text(50, -5, "$\\lambda \\approx -0.075$")
#plt.text(45, -4, "thresh. $ = 20^{\\circ}$")
plt.grid(False)
plt.xticks([0, 25, 50, 75, 100])
plt.yticks([-2, -6, -10])
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(413)
theta = np.linspace(-np.pi, np.pi, 1000)
plt.cla()
plt.bar(np.rad2deg(bb_bins2 - (bb_bins[1] - bb_bins[0])/2.), bb_pdf2, 
        width=np.rad2deg(bb_bins2[1] - bb_bins2[0]),
        edgecolor="none")
#plt.plot(np.rad2deg(bb_bins2), bb_pdf2, lw=4, c='k', marker="o")
#plt.plot(bb_bins, bb_pdf, lw=1, c='gray')
plt.plot(np.rad2deg(theta), bb_cauchy_dist.pdf(theta), c='royalblue', lw=2)
plt.plot(np.rad2deg(theta), bb_vm_dist.pdf(theta), c='r', lw=2)
plt.xlim(-100, 100)
plt.axvline(turn_threshold, c='b', lw=0.3)
plt.axvline(-turn_threshold, c='b', lw=0.3)
plt.xticks([-90, 0, 90])
plt.yticks([0, 1, 2, 3])
plt.xlabel("body bend ($^{\\circ}$)", fontsize=fontsize)
plt.ylabel("prob. dens.", fontsize=fontsize)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "paper_analysis.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


################################################################################################

t_anomdiff, MSD_anomdiff = MSD_i(t, COMs, len(MSD)/20)
plt.ioff()
fig = plt.figure("anomalous diffusion analysis", figsize=(4, 4))
plt.clf()
plot = fig.add_subplot(111)
plt.loglog(t_anomdiff, MSD_anomdiff/10000., c='k', lw=5)
plt.plot(t, (MSD_fit_slope*t)/10000., lw=2, c='r')
plt.plot(t, 0.85*10**-3*t**2, lw=2, c='royalblue')
plt.xlabel("t (s)", fontsize=fontsize)
plt.ylabel("$\\langle d^2 \\rangle$/t (segs$^{2} \\times 10^4$)", fontsize=fontsize)
#plot = fig.add_subplot(133)
#plt.loglog(t_anomdiff[1:], (0.001*t_anomdiff[1:]/10000.)/t_anomdiff[1:], c='b', lw=2)
#plt.loglog(t_anomdiff[1:], (t_anomdiff[1:]**2)/t_anomdiff[1:], c='b', lw=2)
#plt.loglog(t_anomdiff[1:], (MSD_anomdiff[1:]/10000.)/t_anomdiff[1:], c='k', lw=5)
#plt.plot(t, (MSD_fit_slope*t/t)/10000., lw=2, c='r')
#plt.plot(t, 0.55*10**-3*t, lw=2, c='g')
#plt.ylim(10**-8, 1000)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "anomalous_diffusion_analysis.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()

################################################################################################

plt.ioff()
fig = plt.figure("speed-curvature power law", figsize=(4, 4))
plt.clf()
plot = fig.add_subplot(111)
plt.scatter(np.log10(np.concatenate(Cs))[::250], 
            np.log10(np.concatenate(As))[::250], c='grey', edgecolors="none",
            s=3,
            alpha=1.)
plt.plot(np.log10(C_axis), CA_fit_high_C_slope*np.log10(C_axis) + CA_fit_high_C_intercept,
        c='RoyalBlue', lw=2, alpha=0.5)
#plt.plot(np.log10(C_axis), CA_fit_low_C_slope*np.log10(C_axis) + CA_fit_low_C_intercept, 
#        c='r', lw=2, alpha=0.3)
plt.xlabel("log$_{10}$ curvature", fontsize=fontsize)
plt.ylabel("log$_{10}$ angular speed", fontsize=fontsize)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.grid(False)
plt.xlim(-6, 6)
plt.ylim(-8, 4)
plt.xticks([-6, -4, -2, 0, 2, 4, 6])
plt.yticks([-7, -2, 3])
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "speed_curvature_power_law.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


plt.ioff()
fig = plt.figure("speed-curvature power law", figsize=(4, 4))
plt.clf()
plot = fig.add_subplot(111)
#plt.scatter(np.log10(np.concatenate(Cs))[::250], 
#            np.log10(np.concatenate(As))[::250], c='grey', edgecolors="none",
#            s=3,
#            alpha=1.)
plt.scatter(np.log10(np.concatenate(Cs))[::100], 
            np.log10(np.concatenate(As))[::100], c='grey', edgecolors="none",
            s=2,
            alpha=0.1)
plt.plot(np.log10(C_axis), CA_fit_high_C_slope*np.log10(C_axis) + CA_fit_high_C_intercept,
        c='RoyalBlue', lw=2, alpha=0.5)
#plt.plot(np.log10(C_axis), CA_fit_low_C_slope*np.log10(C_axis) + CA_fit_low_C_intercept, 
#        c='r', lw=2, alpha=0.3)
plt.xlabel("log$_{10}$ curvature", fontsize=fontsize)
plt.ylabel("log$_{10}$ angular speed", fontsize=fontsize)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.grid(False)
plt.xlim(-6, 6)
plt.ylim(-8, 4)
plt.xticks([-6, -4, -2, 0, 2, 4, 6])
plt.yticks([-7, -2, 3])
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "speed_curvature_power_law.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()

plt.ioff()
fig = plt.figure("speed-curvature time series", figsize=(3, 4))
plt.clf()
i_repr = 5
plot = fig.add_subplot(211)
plt.ylabel("curvature", fontsize=fontsize)
plt.plot(t[1502:] - t[1502], Cs[i_repr], c='k', lw=2, alpha=0.8)
plt.xlim(0, 120)
plt.xticks([0, 30, 60, 90, 120])
plt.ylim(0, 0.25)
plt.yticks([0, 0.1, 0.2])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plot = fig.add_subplot(212)
plt.ylabel("angular speed", fontsize=fontsize)
plt.xlabel("time (s)")
plt.plot(t[1502:] - t[1502], As[i_repr], c='k', lw=2, alpha=0.8)
plt.xlim(0, 120)
plt.xticks([0, 30, 60, 90, 120])
plt.ylim(0, 0.25)
plt.yticks([0, 0.1, 0.2])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
#plt.xticks([-6, -4, -2, 0, 2, 4, 6])
#plt.yticks([-7, -2, 3])
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "speed_curvature_time_series.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


"""
Chaos analysis etc.
"""

# lce analysis
lce_analysis = np.load(LCE_ANALYSIS_PATH + "lce_analysis.npy")

# representative data
sr, x = np.load(DATA_PATH + data_files[0])      # load sampling rate and state trajectory
m_x = x[:, N_neuron:(N_neuron + 2*N_DOF)]       # slice out mechanical state
q = m_x[:, :N_DOF]                              # slice out mechanical configuration
q_vec = q.reshape(len(t), -1, 2)                # configuration as [x, y] vectors
lengths = np.linalg.norm(np.diff(q_vec, axis=1), axis=2)
length_rates = np.diff(lengths, axis=0)
angles = np.diff(np.arctan2(np.diff(q_vec, axis=1)[:, :, 0],
                            np.diff(q_vec, axis=1)[:, :, 1]), axis=1)
angles = np.unwrap(angles, axis=0)   


plt.ioff()
t0 = 14
tF = 24
fig = plt.figure("internal configuration", figsize=(8, 3))
plt.clf()
plot = fig.add_subplot(121)
plt.plot(t - t0, lengths + np.arange(11), lw=2)
plt.ylabel("axial stretches (segs)")
plt.xlabel("time (s)")
plt.xlim(0, tF - t0)
plt.ylim(0, 12)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

t0 = 13
tF = 23
angle_scale = 0.7
plot = fig.add_subplot(122)
plt.plot(t - t0, angle_scale*angles + np.arange(10), lw=2)
plt.ylabel("transverse bends (rad)")
plt.xlabel("time (s)")
plt.xlim(0, tF - t0)
plt.ylim(0, 10)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "internal_configuration.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()

##########################################

plt.ioff()
fig = plt.figure("chaos analysis", figsize=(8, 3))
plt.clf()
plot = fig.add_subplot(236)
plt.ylabel("MLCE")
plt.plot(lce_analysis[1], lw=2, c='k')
plt.axhline(lce_analysis[0][0], lw=4, c='b', alpha=0.2)
plt.text(200, 16, "MLCE = " + str(np.round(lce_analysis[0][0])) + " bits s$^{-1}$")
plt.yticks([0, 10, 20])
plt.xlabel("iteration")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(233)
plt.bar(full_state_D_c_bins[:-1], full_state_D_c_pdf[0],
        width=full_state_D_c_bin_width, edgecolor="none")
#plt.axvline(full_state_D_c_mean, lw=2, c='r')
plt.axvline(full_state_D_c_median, lw=2, c='b')
#plt.axvline(full_state_D_c_mode, lw=2, c='g')
plt.xlabel("correlation dimension")
plt.ylabel("prob. dens.")
plt.xticks([1, 2, 3, 4])
plt.xlim(1, 4)
plt.yticks([0, 1, 2, 3])
plt.ylim(0, 3)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

n_plot = 100
plot = fig.add_subplot(231)
[plt.plot(f, np.log(psd), c='k', lw=0.025) for f, psd in 
            zip(f_tail_seg_psds[:n_plot], tail_seg_psds[:n_plot])]
plt.ylim(-10, 10)
plt.xlim(0, 10)
plt.ylabel("log PSD $q$")
plt.yticks([-10, 0, 10])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(234)
[plt.plot(f, np.log(psd), c='k', lw=0.025) for f, psd in 
            zip(f_head_seg_psds[:n_plot], head_seg_psds[:n_plot])]
plt.ylim(-10, 10)
plt.xlim(0, 10)
plt.ylabel("log PSD $\phi$")
plt.xlabel("frequency (Hz)")
plt.yticks([-10, 0, 10])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(232)
[plt.plot(t_lag, corr/np.max(corr), c='k', lw=0.005) for t_lag, corr in 
            zip(t_lags_tail_seg_corrs[:n_plot], tail_seg_corrs[:n_plot])]
plt.cla()
plt.plot(t_lag, corr/np.max(corr), c='k', lw=2)
plt.ylabel("autocorr. $q$")
plt.ylim(-1, 1)
plt.yticks([-1, 0, 1])
plt.xticks([-60, -30, 0, 30, 60])
plt.xlim(-70, 70)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(235)
[plt.plot(t_lag, corr/np.max(corr), c='k', lw=0.005) for t_lag, corr in 
            zip(t_lags_head_seg_corrs[:n_plot], head_seg_corrs[:n_plot])]
plt.cla()
plt.plot(t_lag, corr/np.max(corr), c='k', lw=2)
plt.ylabel("autocorr. $\phi$")
plt.xlabel("time lag (s)")
plt.xticks([-60, -30, 0, 30, 60])
plt.ylim(-1, 1)
plt.yticks([-1, 0, 1])
plt.xlim(-70, 70)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "chaos_analysis.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


##########################################

plt.ioff()
fig = plt.figure("neuromechanical state", figsize=(4, 8))
plt.clf()
plot = fig.add_subplot(511)
plt.plot(t - t0, lengths + np.arange(N_seg), c='k', lw=2)
plt.ylabel("stretch (segs)")
plt.xlim(0, tF - t0)
plt.ylim(-1, 12)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(512)
plt.plot(t[:-1] - t0, length_rates + np.arange(N_seg), c='k', lw=2)
plt.ylabel("stretch rate (segs s$^{-1}$)")
plt.xlim(0, tF - t0)
plt.ylim(-1, 12)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(513)
plt.plot(t - t0, x[:, :N_seg] + np.arange(N_seg), c='k', lw=2)
plt.ylabel("SN")
plt.xlim(0, tF - t0)
plt.ylim(-1, 12)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(514)
plt.plot(t - t0, x[:, N_seg:2*N_seg] + np.arange(N_seg), c='k', lw=2)
plt.ylabel("IN")
plt.xlim(0, tF - t0)
plt.ylim(-1, 12)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(515)
plt.plot(t - t0, x[:, 2*N_seg:3*N_seg] + np.arange(N_seg), c='k', lw=2)
plt.ylabel("MN")
plt.xlim(0, tF - t0)
plt.ylim(-1, 12)
plt.xlabel("time (s)")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "neuromech_state.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


##########################################


plt.ioff()
fig = plt.figure("two-segment analysis", figsize=(8, 4))
plt.clf()
t_min = 20; t_max = 30

plot = fig.add_subplot(231)
plt.plot(t_concat, np.concatenate(tail_speeds), c='DarkBlue', lw=2)
plt.xlim(t_min, t_max)
ylim = max(np.abs(ts_min), np.abs(ts_max))
plt.ylim(0, ylim)
plt.ylabel("$v$ (segs $s^{-1}$)")
plt.yticks([0, 0.5, 1])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(234)
plt.plot(t_concat, np.concatenate(bending_velocities), c='DarkRed', lw=2)
plt.xlim(t_min, t_max)
ylim = max(np.abs(av_min), np.abs(av_max))
plt.ylim(-ylim, ylim)
plt.ylabel("$\\nu$ (rad $s^{-1}$)")
plt.xlabel("time (s)")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(232)
plt.plot(ts_pdf, ts_bins, lw=2, c='DarkBlue')
ylim = max(np.abs(ts_min), np.abs(ts_max))
plt.ylim(0, ylim)
plt.yticks([0, 0.5, 1])
plt.xlabel("probability density")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(235)
plt.plot(np.log(av_pdf), av_bins, lw=2, c='DarkRed')
ylim = max(np.abs(av_min), np.abs(av_max))
plt.ylim(-ylim, ylim)
plt.xlabel("log probability density")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(233)
for f, S_ts in zip(f_tail_speed_psds, np.log(tail_speed_psds)) :
    plt.plot(f, S_ts, c='DarkBlue', lw=0.1)
plt.plot(f_mean_tail_speed_psd, np.log(mean_tail_speed_psd), c='b', lw=2)
plt.xlim(0, 4)
plt.ylim(-10, 10)
plt.ylabel("log PSD$\\left[ v \\right]$")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(236)
for f, S_av in zip(f_bending_velocities_psds, np.log(bending_velocities_psds)) :
    plt.plot(f, S_av, c='DarkRed', lw=0.1)
plt.plot(f_mean_bending_velocities_psd, np.log(mean_bending_velocities_psd), c='r', lw=2)
plt.xlim(0, 4)
plt.ylim(-10, 10)
plt.ylabel("log PSD$\\left[ \\nu \\right]$")
plt.xlabel("frequency (Hz)")
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
nm.util.hide_spines()
plt.tight_layout()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "two_segment_analysis__thin.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()




"""
Exploration animations.
"""

# set animation parameters
n_sims = 50                 # number of simulation runs to plot simultaneously
i0 = 0                      # initial timestep
iF = -1                     # final timestep
step = 4                    # downsampling step (i.e. take a sample every "step" interval)
ax_lim = 500                # axis limits
dpi = 300                   # dots per inch
figsize= 6                  # figure size in inches

# load data from local directory
dats = []
for i in xrange(len(data_files[:n_sims])) :
    print "loading " + data_files[:n_sims][i] + " (" + str(i + 1) + " of " + str(n_sims) + ")"
    dats.append(np.load(DATA_PATH + data_files[:n_sims][i])[1][i0:iF:step, tx_i:h_py_i + 1])
dats = np.array(dats)

# compute centre of mass trajectories
R_xs = np.array([np.mean(x[:, :24:2], axis=1) for x in dats])
R_ys = np.array([np.mean(x[:, 1:24:2], axis=1) for x in dats])

# useful views on data
frames = dats[0][:, :24]
COM_x = R_xs[0]
COM_y = R_ys[0]

# some useful quantities
R = lambda theta : sy.Matrix([[sy.cos(theta), -sy.sin(theta)], [sy.sin(theta), sy.cos(theta)]])
D_axial = sp.linalg.circulant([1] + [0]*(2*N - 3) + [-1, 0])[:-2]
rot = np.array([[0, 1], [-1, 0]])
seg_widths = 0.5*np.cos(np.linspace(-1.5, 2, N)) + 0.6

# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg'] # NOTE use on laptop
#Writer = animation.writers['imagemagick'] # NOTE use on work computer
Writer = animation.writers['mencoder']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure("animation", figsize=(figsize, figsize))
plt.clf()
plt.grid(True)

colors = ["Gray", "Red", "Peru", "Orange", "LimeGreen", "ForestGreen",
            "MediumTurquoise", "Navy", "Gray", "DimGray", "Black"]
#colors = ["Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray", "Gray",
#            "Gray", "Gray", "Gray"]

LAB_plot_index = 111

# set up LAB frame plots
plt.subplot(111)
#plt.plot(frames[:, 0], frames[:, 1], c='blue', lw=2, alpha=0.1)
#plt.plot(frames[:, -2], frames[:, -1], c='pink', lw=2, alpha=0.4)
#plt.plot(COM_x, COM_y, c='gray', alpha=0.2, lw=2)

segs_LAB = []
line_LAB = []
scats_LAB = []
for i in xrange(len(dats)) :
    segs_LAB.append([plt.plot([], [], '-', lw=0.7, zorder=2, c=colors[i % len(colors)])[0] 
            for i in xrange(12)])
    line_LAB.append(plt.plot([], [], '-', lw=1, zorder=1, c='gray')[0])
    scats_LAB.append([plt.plot([], [], '-', lw=2, zorder=3, c=colors[i % len(colors)])[0] 
            for i in xrange(12)])

# LAB frame limits
LAB_window_width = 200
LAB_mid_x = 0
LAB_mid_y = 0

#plt.xlim(LAB_mid_x - LAB_window_width, LAB_mid_x + LAB_window_width)
#plt.ylim(LAB_mid_y - LAB_window_width, LAB_mid_y + LAB_window_width)
int_ax_lim = int(ax_lim)
plt.xticks([-int_ax_lim, -int_ax_lim/2, 0, int_ax_lim/2, int_ax_lim])
plt.yticks([-int_ax_lim, -int_ax_lim/2, 0, int_ax_lim/2, int_ax_lim])
plt.xlim(-ax_lim, ax_lim)
plt.ylim(-ax_lim, ax_lim)
plt.tight_layout()


def update_line(i) :
    # extract current state frame, and then extract positions from this
    frms = dats[:, i][:, :24]

    # draw back-bone in COM and LAB frames
    for line_LAB_i, frm_i in zip(line_LAB, frms) :
        line_LAB_i.set_data(frm_i[::2], frm_i[1::2])

    # draw coloured dots for segment boundary COMs, in COM and LAB frames
    for scats_LAB_i, frm_i in zip(scats_LAB, frms) :
        for scat_LAB, x, y in zip(scats_LAB_i, frm_i[::2], frm_i[1::2]) :
            scat_LAB.set_data([x], [y])

    # find segment boundary vectors (normal to backbone)
    for i in xrange(len(frms)) :
        pos_vecs = np.array([frms[i][::2], frms[i][1::2]]).T

        tangents = np.dot(D_axial, frms[i])
        tangent_vecs = np.array([tangents[::2], tangents[1::2]]).T
        tangent_vecs_lengths = np.linalg.norm(tangent_vecs, axis=1)
        unit_tangent_vecs = np.array([vec/length for vec, length in zip(tangent_vecs, tangent_vecs_lengths)])
    
        tail_nrm_vec = np.dot(rot, unit_tangent_vecs[0])
        tail_nrm_vec = tail_nrm_vec/np.linalg.norm(tail_nrm_vec)
        tail_nrm_vec = seg_widths[0]*tail_nrm_vec
        body_nrm_vecs = np.diff(unit_tangent_vecs, axis=0)
        body_nrm_vecs = np.array([vec/length for vec, length in zip(body_nrm_vecs,
                                                        np.linalg.norm(body_nrm_vecs, axis=1))])
        head_nrm_vec = np.dot(rot, unit_tangent_vecs[-1])
        head_nrm_vec = head_nrm_vec/np.linalg.norm(head_nrm_vec)
        head_nrm_vec = seg_widths[-1]*head_nrm_vec
    
        # draw segment boundaries in COM and LAB frames
        segs_LAB[i][0].set_data([pos_vecs[0][0] + tail_nrm_vec[0], pos_vecs[0][0] - tail_nrm_vec[0]], 
                             [pos_vecs[0][1] + tail_nrm_vec[1], pos_vecs[0][1] - tail_nrm_vec[1]])
        segs_LAB[i][-1].set_data([pos_vecs[-1][0] + head_nrm_vec[0], pos_vecs[-1][0] - head_nrm_vec[0]], 
                              [pos_vecs[-1][1] + head_nrm_vec[1], pos_vecs[-1][1] - head_nrm_vec[1]])
        for body_seg, nrm_vec, pos_vec, length in zip(segs_LAB[i][1:-1], body_nrm_vecs, 
                                                        pos_vecs[1:-1], seg_widths[1:-1]) :
            body_seg.set_data([pos_vec[0] + length*nrm_vec[0], pos_vec[0] - length*nrm_vec[0]], 
                              [pos_vec[1] + length*nrm_vec[1], pos_vec[1] - length*nrm_vec[1]])
    if i <= 10 :
        plt.tight_layout()
    return 1,


plt.xlabel("")
plt.ylabel("")
plt.title('')
#line_ani = animation.FuncAnimation(fig, update_line, frames=len(frames) - 1, interval=10, blit=False)
line_ani = animation.FuncAnimation(fig, update_line, frames=len(frames) - 1,
        interval=10, blit=False)

line_ani.save(PLOT_PATH + "diffusing_population.mp4", writer=writer, dpi=dpi)
plt.close()
