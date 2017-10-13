#! /usr/bin/env python2

import time
import itertools as it
import sympy as sy
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

import util


"""
Dynamics analysis functions.
"""

def lce_estimate(x0, f, p0, t_step=0.005, pb_step=100, n_pb=500, n_ic_steps=15000,
        d0=10**-7, tol=0.0001, debug=False, n_pb_discard=10,
        dist=np.linalg.norm, log=np.log, timescale=1, mask=None, E=None) :
    """
    Estimate the maximal Lyapunov characteristic exponent (lce) for the dynamical
    system described by f.
    """
    # TODO allow for deterministic execution (e.g. specify seed for
    # np.random.random OR allow user to set rand variable explicitly
    # TODO allow for termination on satisfaction of a convergence condition

    N = len(x0)
    if mask is None :
        mask = np.ones(len(x0), dtype=np.bool)

    def _lce_step(x0, y0, t_step_arr, f, p0, tol, E=E) :
        # integrate the dynamics f, forwards in time, from x0 and y0 to final
        # states xF and yF 
        x_arr = util.FORTRAN_integrate(t_step_arr, x0, f, p0, atol=tol, rtol=tol)[:, :N]
        y_arr = util.FORTRAN_integrate(t_step_arr, y0, f, p0, atol=tol, rtol=tol)[:, :N]
        xF = x_arr[-1]
        yF = y_arr[-1]

        # calculate initial separation d0 and final separation dF
        d0 = dist(x0[mask] - y0[mask])
        dF = dist(xF[mask] - yF[mask])

        # calculate a finite-time estimate of the maximal LCE
        ft_lce = log(dF/d0)/(timescale*t_step_arr[-1] - timescale*t_step_arr[0])

        # plot results of this step if debugging is enabled
        if debug : 
            plt.subplot(121)
            plt.plot(t_step_arr*timescale, log(dist(x_arr[:, mask] - y_arr[:, mask], axis=1)), c='grey')
            plt.xlabel("$t$")
            plt.ylabel("$\\textrm{log} \:\: \delta w$")
            plt.ylim(log(d0), log(d0 + 10**-3))
            plt.tight_layout()
        del(x_arr, y_arr)
    
        return ft_lce, xF, yF

    if debug : print "integrating from ICs..."
    # integrate from given initial conditions for n_ic_steps time steps (we are
    # hoping to hit an attractor); then reset initial conditions
    t_ic_arr = np.linspace(0, n_ic_steps*t_step, n_ic_steps)
    x0 = util.FORTRAN_integrate(t_ic_arr, x0, f, p0, rtol=tol, atol=tol)[-1][:N]

    if debug : print "generating random nearby IC..."
    # randomly generate a nearby initial condition, separated from x0 by a
    # distance d0
    rand = 2*(np.random.random(N) - 0.5)
    rand_mask = d0*rand[mask]/np.linalg.norm(rand[mask])
    rand[mask] = rand_mask
    y0 = np.copy(x0)
    y0[mask] = x0[mask] + rand[mask]

    if debug : print "calculating finite-time LCE estimates..."
    # array to hold finite-time lce calculated at each pullback
    ft_lce = []

    # integrate the dynamics with repeated pullbacks of the nearby trajectory;
    # estimate the lce at each pullback
    t0 = t_ic_arr[-1]
    for i in xrange(n_pb) :
        print "integrating pullback " + str(i + 1) + " of " + str(n_pb)
        # construct time axis
        t_pb_arr = np.linspace(t0 + i*pb_step*t_step, t0 + (i + 1)*pb_step*t_step, pb_step)
#        t_pb_arr = np.arange(t0 + i*pb_step*t_step, t0 + (i + 1)*pb_step*t_step, t_step)

        # integrate the dynamics and store the calculated finite-time lce
        ft_lce_i, xF, yF = _lce_step(x0, y0, t_pb_arr, f, p0, tol)
        ft_lce.append(ft_lce_i)

        # pullback the nearby trajectory to a distance d0 along the direction
        # given by (yF - xF)
        x0 = xF
        y0 = np.copy(xF)
        y0[mask] = xF[mask] + d0*(yF[mask] - xF[mask])/dist(yF[mask] - xF[mask])

        # calculate a better estimate for the true maximal lce by averaging
        # over the finite time estimates
        lce_estimate = np.mean(ft_lce[n_pb_discard:])
        ft_lce_loc = np.array(ft_lce)
        lce_estimate = np.mean(ft_lce_loc[np.abs(ft_lce_loc) != np.inf][n_pb_discard:])

        # calculate energy
        if E is not None :
            print "current LCE estimate = " + str(lce_estimate) + " (i = " + str(i) + " of " + str(n_pb) + ", t = " + str(t_pb_arr[-1]*timescale) + ", E = " + str(E(xF[:len(x0)])[0]) + ")"
        else : 
            print "current LCE estimate = " + str(lce_estimate) + " (i = " + str(i) + " of " + str(n_pb) + ", t = " + str(t_pb_arr[-1]*timescale) + ")"
        if debug :
            plt.subplot(122)
            plt.cla()
#            plt.scatter(i, np.mean(lce_estimate), s=10, c='grey', edgecolors="none")
            lce_estimate_arr = np.array([np.mean(ft_lce_loc[np.abs(ft_lce_loc) != np.inf][:i]) for i in xrange(len(ft_lce_loc))])
            plt.plot(lce_estimate_arr)
            if len(lce_estimate_arr) > 2 :
                plt.ylim(0, np.max(lce_estimate_arr[1:]))
            plt.xlabel("pullback")
            plt.ylabel("$\lambda$")
            plt.pause(0.01)

        del(t_pb_arr, ft_lce_loc)

    # calculate a best estimate for the true maximal LCE by averaging over the
    # finite time estimates (the initial few finite time estimates may be
    # discarded to ensure the pullback trajectory had converged to the
    # direction with the maximal LCE)
    ft_lce = np.array(ft_lce)
    lce = np.mean(ft_lce[np.abs(ft_lce) != np.inf][n_pb_discard:])
    lce_estimate_arr = np.array([np.mean(ft_lce[np.abs(ft_lce) != np.inf][n_pb_discard:i]) for i in xrange(len(ft_lce[np.abs(ft_lce) != np.inf]) - n_pb_discard)])

    return [lce], lce_estimate_arr, ft_lce


"""
Data analysis functions -- frequency domain.
"""

def fft(t, x, timescale=0.001, detrend=True, axis=0) :
    # prepare frequency axis
    sample_length = t.shape[0]
    dt = (t[1] - t[0])*timescale
    f = sp.fftpack.fftfreq(sample_length, dt)

    # detrend data
    if detrend : x = sp.signal.detrend(x, axis=axis)

    # compute fft, magnitude, and phase
    X = sp.fftpack.fft(x, axis=axis)
    X_mag = np.abs(X)
    X_phase = np.angle(X)

    # shift to convenient ordering
    f = sp.fftpack.fftshift(f)
    X = sp.fftpack.fftshift(X)
    X_mag = sp.fftpack.fftshift(X_mag)
    X_phase = sp.fftpack.fftshift(X_phase)
    return f, X, X_mag, X_phase


def psd(t, x, timescale=0.001, detrend=True, axis=0) :
    f, X, X_mag, X_phase = fft(t, x, timescale=timescale, detrend=detrend, axis=axis)
#    S = 2*(1./t[-1])*X_mag**2
    S = ((t[1] - t[0])**2*X_mag**2)/t[-1]
    return f, S


def extract_strongest_frequency(t, x, timescale=0.001, axis=0) :
    f, X, X_mag, X_phase = fft(t, x, timescale=timescale, axis=axis)
    f0_i = np.argmax(X_mag, axis=axis)
    f0 = f[f0_i]
    return f0


def extract_phase_shift(t, x, timescale=0.001, axis=0) :
    f, X, X_mag, X_phase = fft(t, x, timescale=timescale, axis=axis)
    f0_i = np.argmax(X_mag, axis=axis)
    theta = np.rad2deg(X_phase[f0_i])
    return theta


def boxcar_kernel(dur=0.5, dt=1., timescale=0.001) :
    M = (dur/timescale)/dt
    kernel = sp.signal.windows.boxcar(M, sym=True)
    kernel = kernel/np.sum(kernel)
    return kernel


def gaussian_kernel(dur=0.5, dt=1., timescale=0.001) :
    M = (dur/timescale)/dt
    kernel = sp.signal.windows.gaussian(M, std=M/6., sym=True)
    kernel = kernel/np.sum(kernel)
    return kernel


def correlation(t, x, y) :
    corr = sp.signal.fftconvolve(x, y[::-1], mode="same")
    dt = t[1] - t[0]
    t = np.linspace(-len(corr)*dt/2., len(corr)*dt/2., len(corr))
    return t, corr


def phase_lag_correlation_estimate(t, x, y) :
    tcorr, Xcorr = correlation(t, x, y)
    f, Xfft, Xfft_mag, Xfft_phase = fft(tcorr, Xcorr)
    f0_i = np.argmax(Xfft_mag)
    phase_lag = np.rad2deg(Xfft_phase[f0_i + 1])
    return phase_lag


from scipy.signal import butter, lfilter

def butterworth_bandpass(t, x, low_cut=0.0, high_cut=10.0, order=2) :
    nyq = 0.5/(t[1] - t[0])
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')

    y = lfilter(b, a, x)
    return y


def butterworth_bandpass_zero_phase(t, x, low_cut=0.0, high_cut=10.0, order=2) :
    y1 = butterworth_bandpass(t, x, low_cut, high_cut, order)
    y2 = butterworth_bandpass(t, y1[::-1], low_cut, high_cut, order)[::-1]
    return y2


"""
Data analysis functions -- 2D model behaviour.
"""

def run_lengths(t, bb, threshold=30, minimum_length=0.5) : 
#    bb_abs_deg = np.rad2deg(np.abs(bb))
#    casting_intervals = np.diff(t[bb_abs_deg > threshold])
#    run_lengths = t[bb_abs_deg > threshold][:-1][casting_intervals > minimum_length]

    intervals = np.diff(t[np.rad2deg(np.abs(bb)) > threshold])
    run_lengths = intervals[intervals > minimum_length]
    return run_lengths

def tail_speed(x, t_px_i=24, t_py_i=25) :
    """
    Compute the tail speed.
    """
    ts = np.linalg.norm(np.array([x[:, t_px_i],
                                  x[:, t_py_i]]), axis=0)
    return ts

def body_bend(x, tx_i=0, ty_i=1, mx_i=12, my_i=13,
        hx_i=22, hy_i=23) :
    """
    Compute the angle between the head and the body axis in radians.
    """
    # take radius vectors pointing from midpoint to tail
    r_t = np.array([x[:, tx_i] - x[:, mx_i],
                    x[:, ty_i] - x[:, my_i]]).T

    # take radius vectors pointing from midpoint to head
    r_h = np.array([x[:, hx_i] - x[:, mx_i],
                    x[:, hy_i] - x[:, my_i]]).T

    # construct head and tail unit vectors
#    n_t = (r_t.T/np.linalg.norm(r_t, axis=1)).T
#    n_h = (r_h.T/np.linalg.norm(r_h, axis=1)).T

    # compute angle between head and cartesian axes
    h_angle = np.arctan2(r_h[:, 1], r_h[:, 0])

    # compute angle between negative tail and cartesian axes
    t_angle = np.arctan2(-r_t[:, 1], -r_t[:, 0])

    # compute angle between negative tail and head (body bend)
    body_bend = h_angle - t_angle
    return np.unwrap(body_bend)


def head_angular_velocity(x, mx_i=12, my_i=13,
        hx_i=22, hy_i=23, h_px_i=46, h_py_i=47) :
    """
    Compute the angular velocity of the head relative to the midpoint.
    """
    # take radius vectors pointing from midpoint to head
    r_h = np.array([x[:, hx_i] - x[:, mx_i],
                    x[:, hy_i] - x[:, my_i]]).T

    # find velocity vector of the head
    v_h = np.array([x[:, h_px_i], x[:, h_py_i]]).T

    # compute cross-product of the radius and velocity vector
    r_x_v = np.cross(r_h, v_h)

    # normalise by the square of the radius vector to find angular velocity
    w = r_x_v/np.linalg.norm(r_h, axis=1)**2

    return w


def odor_bearing_circular_symmetric(x, mx_i=12, my_i=13, tx_i=0, ty_i=1) :
    """
    Calculate odor bearing assuming circular symmetric odor gradient centred on
    origin of coordinate system.
    """
    # take radius vectors pointing from tail to midpoint
    r_t = np.array([x[:, mx_i] - x[:, tx_i],
                    x[:, my_i] - x[:, ty_i]]).T

    # take radius vectors pointing from origin to midpoint
    r_m = np.array([x[:, mx_i], x[:, my_i]]).T

    # compute angle between tail vector and cartesian axes
    body_angle = np.arctan2(r_t[:, 1], r_t[:, 0])

    # compute angle between gradient vector and cartesian axes
    gradient_angle = np.arctan2(-r_m[:, 1], -r_m[:, 0])

    bearing_angle = gradient_angle - body_angle
    return bearing_angle


def head_sweep_durations(t, x, mx_i=12, my_i=13, hx_i=22, hy_i=23, h_px_i=46,
        h_py_i=47, epsilon=0.001) :
    """
    Calculate durations of head sweeps, defined by successive zero-crossings of
    the head angular velocity.
    """
    bending_velocities = head_angular_velocity(x, mx_i, my_i, hx_i, hy_i,
            h_px_i, h_py_i)
    zero_crossing_mask = np.diff(bending_velocities < epsilon)
    durations = np.diff(t[:-1][zero_crossing_mask])
    return durations


def head_sweep_amplitudes(t, x, mx_i=12, my_i=13, hx_i=22, hy_i=23, h_px_i=46,
        h_py_i=47, epsilon=0.001) :
    """
    Calculate durations of head sweeps, defined by successive zero-crossings of
    the head angular velocity.
    """
    bending_velocities = head_angular_velocity(x, mx_i, my_i, hx_i, hy_i,
            h_px_i, h_py_i)
    zero_crossing_mask = np.diff(bending_velocities < epsilon)
    sweep_indices = np.arange(len(zero_crossing_mask))[zero_crossing_mask]

    sweep_amps = []
    for i in xrange(len(sweep_indices) - 1) :
        sweep = bending_velocities[sweep_indices[i]:sweep_indices[i + 1]]
        sweep_amp = np.max(np.abs(sweep))
        sweep_amps.append(sweep_amp)
    sweep_amps = np.array(sweep_amps)
    return sweep_amps


def curvature_angular_speed_analysis(t, COM, discard=1500) :
    dt = t[1] - t[0]
    i = int(discard/2.)

    x = butterworth_bandpass_zero_phase(t, COM[:, 0], high_cut=0.07, order=2)[i:-i]
    y = butterworth_bandpass_zero_phase(t, COM[:, 1], high_cut=0.07, order=2)[i:-i]
    Dx = np.diff(x)/dt
    Dy = np.diff(y)/dt
    DDx = np.diff(Dx)/dt
    DDy = np.diff(Dy)/dt

    C = np.abs(Dx[:-1]*DDy - Dy[:-1]*DDx)/(Dx[:-1]**2 + Dy[:-1]**2)**(3./2.)
    A = np.abs(np.diff(np.unwrap(np.arctan2(Dy, Dx)))/dt)

    return t[i:-i], C, A
    

"""
Data analysis functions -- dimensionality estimation.
"""

def correlation_dimension_estimate(x, e_pts=100, debug=False) :
    """
    Estimate the dimension of a fractal dataset using the correlation
    dimension. If the average number of data points within a ball of radius e
    centred on a particular data point is C(e), then the correlation dimension
    is found as the exponent by which C(e) scales with e, i.e.

        C(e) ~ a*exp(D*e)

    where D is the correlation dimension.
    """
    # find the distances between all points in the dataset
    dist = sp.spatial.distance.cdist(x, x)

    # define the function C(e) which gives the average number of datapoints
    # within a ball of radius e centred on a given datapoint
    C = np.vectorize(lambda e : np.mean(np.sum(dist < e, axis=0)))

    # estimate exponential scaling of C(e) from a log-log plot
    e_axis = np.linspace(1, np.max(dist), e_pts)
    C_num = C(e_axis)
    lr = sp.stats.linregress(np.log(e_axis)[:10], 
                             np.log(C_num)[:10])
    D = lr[0]

    if debug :
        plt.figure("correlation dimension")
        plt.title("correlation dimension estimate debugging")
        plt.plot(np.log(e_axis), np.log(C_num))
        plt.plot(np.log(e_axis), D*np.log(e_axis) + lr[1])

    return D


def box_counting_dimension_estimate_2d(dat, min_scale=None, max_scale=None,
        num_scales=20,
        debug=False) :
    """
    Estimate the fractal dimension of a 2D dataset using the box-counting
    dimension.
    """

    x_min = np.min(dat[:, 0])
    x_max = np.max(dat[:, 0])
    y_min = np.min(dat[:, 1])
    y_max = np.max(dat[:, 1])

    if min_scale is None : min_scale = 1
    if max_scale is None : max_scale = min(np.log2(x_max - x_min),
                                           np.log2(y_max - y_min))
    
    if debug : print x_min, x_max, y_min, y_max, min_scale, max_scale
    
    # compute the fractal dimension considering only scales in a logarithmic list
    scales=np.logspace(min_scale, max_scale, num=num_scales, endpoint=False, base=2)
    x_bins = ((x_max - x_min)/scales).astype(np.int)
    y_bins = ((y_max - y_min)/scales).astype(np.int)
    unique_x_bins_index = np.unique(x_bins, return_index=True)[-1]
    unique_y_bins_index = np.unique(y_bins, return_index=True)[-1]
    unique_bins_index = np.unique(np.append(unique_x_bins_index,
                                            unique_y_bins_index))
    scales = scales[unique_bins_index]
    if debug : print "actual num_scales=" + str(len(scales))
    Ns=[]
    # looping over several scales
    for scale in scales:
        if debug : print "======= Scale ", scale
        # computing the histogram
        x_bins = ((x_max - x_min)/scale).astype(np.int)
        y_bins = ((y_max - y_min)/scale).astype(np.int)
        if x_bins <= 0 : x_bins = 1
        if y_bins <= 0 : y_bins = 1
        if debug: print x_bins, y_bins
        H, edges = np.histogramdd(dat, bins=(x_bins, y_bins))
        if debug : print int(x_bins), int(y_bins)
        Ns.append(np.sum(H > 0))
     
    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    D = -coeffs[0]
    if debug : print "The box-counting dimension is", D

    if debug :
        plt.figure("box-counting dimension estimate")
        plt.plot(np.log(scales), np.log(Ns), 'o', mfc='none')
        plt.plot(np.log(scales), np.polyval(coeffs,np.log(scales)))
        plt.xlabel('log $\epsilon$')
        plt.ylabel('log N')
        plt.tight_layout()

    return D
