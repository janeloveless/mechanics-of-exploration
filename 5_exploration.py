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
Simulate the motion of the body in the presence of friction and driving forces.
All analysis of simulation results is done in 6_trajectory_analysis.py to save
space. Note that this script is computationally heavy -- it may take several
hours to run!
"""

"""
Set some parameters. 

Some others will be defined later in the code, where it is more convenient.
"""

# mechanical parameters (more defined later in p0 vector)
N_seg = 11                              # number of segments in mechanical model
N = N_seg + 1                           # number of masses in mechanical model
mech_timescale = 1./1000.               # timescale of mechanical system, relative to neural system

# sensory neuron parameters
SN_q_gain = 0                           # sensory neuron stretch sensitivity
SN_p_gain = 1                           # sensory neuron stretch rate sensitivity

# filesystem parameters (more defined later in p0 vector)
IC_PATH = "./data/initial_conditions/5_exploration/"
OUTPUT_PATH = "./data/output/5_exploration/simulation_outputs/"
LCE_ANALYSIS_PATH = "./data/output/5_exploration/LCE_analysis/"
F_PATH = "./FORTRAN_sources/"


"""
Set up the mechanical system.
"""

print "Setting up mechanical system..."
m = nm.model.SimplePlanarCrawler(N)

print "Defining useful kinematic quantities..."
q = m.x[:2*N]
q_vec = np.array(q).reshape(-1, 2)
q_diffs = np.diff(q_vec, axis=0)
q_lengths = [sy.sqrt(qi) for qi in np.sum(q_diffs**2, axis=1)]

Dq = [sy.diff(qi, t) for qi in q]
Dq_vec = np.array(Dq).reshape(-1, 2)
Dq_lengths = [sy.diff(q_length_i, t) for q_length_i in q_lengths]

Dq_to_p = m.f[:2*N]
Dq_to_p_vec = np.array(Dq_to_p).reshape(-1, 2)
p_lengths = [Dq_length_i.subs(zip(Dq, Dq_to_p)) for Dq_length_i in Dq_lengths]


"""
Set up the neural system.
"""

print "Setting up neural system..."
print "Setting sensory neuron inputs to mechanical outputs..."
SN_u = np.concatenate([q_lengths, p_lengths])               # vector of SN inputs
SN_q_ws = (SN_q_gain*np.eye(N_seg)).tolist()                # stretch weight matrix
SN_p_ws = (-SN_p_gain*np.eye(N_seg)).tolist()               # stretch rate weight matrix
SN_ws = [q_w + p_w for q_w, p_w in zip(SN_q_ws, SN_p_ws)]   # total weight matrix
 
print "Constructing neural model..."
n = nm.model.MechanicalFeedbackAndMutualInhibition(N_seg, SN_u, SN_ws)

print "Setting axial mechanical inputs to motor neuron outputs..."
V_MNs = n.x[2*N_seg:]                               # motor neuron activations
m.f = m.f.subs(zip(m.u[:N - 1], V_MNs))             

print "Setting transverse mechanical inputs to zero..."
v_num = [0]*(N - 2)
m.f = m.f.subs(zip(m.u[N - 1:], v_num))


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
f_src = nm.util.FORTRAN_f(x, f, params, verbose=True)

# save FORTRAN source code for future usage
with open(F_PATH + "5_exploration.f", "w") as src_file :
    src_file.write(f_src)

# load FORTRAN source code
f_src = open(F_PATH + "5_exploration.f").read()

print "Compiling RHS function FORTRAN source code to binary..."
f_f77 = nm.util.FORTRAN_compile(f_src)


"""
Generate n simulation outputs, decimate to low sampling rate, then save.
"""

print "Setting simulation parameters..."

n_sims = 1000                              # number of iterations to run
tF = 1000000                               # simulation duration (arbitrary time units)
step_scale = 0.8                            # set simulation step size (arbitrary units)
t_arr = np.linspace(0, tF, step_scale*tF)   # simulation time axis

print "Determining appropriate decimation scheme..."
dom_axial_freq = float(np.load(IC_PATH + "dom_axial_freq.npy")) # fundamental axial frequency
output_fps = 30                             # output sampling rate (Hz)
t_arr_scaled = t_arr*dom_axial_freq         # scaled time axis so that 1 wave = 1 second
samples_per_wave = np.searchsorted(t_arr_scaled, 1) # number of samples in a wave
decimation_step = samples_per_wave/output_fps   # step needed to achieve 30 fps

print "Setting template initial conditions..."

# set mechanical initial conditions
# ... first load mechanical mode shapes
v_a = np.load(IC_PATH + "axial_modal_ics.npy")            # load axial mode shapes
v_t = np.load(IC_PATH + "transverse_modal_ics.npy")       # load transverse mode shapes

# ... initialise mechanical state vector to zero, then construct a starting
# state vector from low frequency mode shapes (ignoring total translations and
# rotations)
m_x0 = np.zeros(4*N)                   
m_x0[:2*N:2] = np.append(v_a[2], v_a[2][0]) + np.arange(N)
m_x0[1:2*N:2] = + 0.2*v_t[2] - 0.2*v_t[3] - 0.2*v_t[4]
orig_m_x0 = np.copy(m_x0)   # store a copy of this IC

m_epsilon = 0.0000001       # set maximum mechanical noise to be added to template

x0 = len(n.x)*[0] + list(m_x0)

print "Setting model parameters..."
# find total length of larva, given initial conditions
L0 = np.sum(np.linalg.norm(np.diff(m_x0[:2*N].reshape(-1, 2), axis=0), axis=1))

# dissipative parameters
#[0.48]*(N_seg - 4) + [0.09]*4 + \
b_head = 0.09
b_body = 0.48
p0 = [-2.0,
      L0,
      -2.0] + \
     [b_body, b_head, b_head] + [b_body]*(N_seg - 4) + [b_head] + \
     [0.25] + \
     (1*np.array([0.067, 0.033, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 0.67, 0.33])).tolist() + \
     [2, 1000] + \
     (1*0.2*np.array([0.35, 0.25] + [1]*(N_seg - 5) + [0.75, 0.5])).tolist() +\
     [2, 0.9] + \
     [5, 0.2, 100, 100, 5, 5, 5, 5, 5, 5, 5, 0.2] + \
     (0.0*np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])).tolist() + \
     [1.5, 1.5, 4.5, 4.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5] + \
     [0.01]
tol = 0.0006

def _print(s) :
    print s
print "Initial conditions :"
[_print(str(x_x0_i))  for x_x0_i in zip(x, x0)]
print "Parameters : "
[_print(str(p_p0_i))  for p_p0_i in zip(params, p0)]


n_successful = 0
n_error = 0
while n_successful < n_sims :
    try :
        print "attempting simulation run " + str(n_successful + 1) + " of " + \
                str(n_sims) + " (" + str(n_error) + " total aborted run(s) so far)"
        ## generate ICs
        # reinitialise to template IC with additive, uniformly distributed noise
        print "setting ICs..."
        m_noise = (1./np.sqrt(4*N))*m_epsilon*2*(np.random.random(4*N) - 0.5)
        m_x0 = orig_m_x0 + m_noise
        # set neural state to zero and combine with mechanical initial conditions
        x0 = len(n.x)*[0] + list(m_x0)
    
        ## run simulation
        print "running simulation..."
        x_arr = nm.util.FORTRAN_integrate(t_arr, x0, f_f77, p0, rtol=tol, atol=tol)
    
        ## decimate 
        print "time-scaling and decimating simulation output..."
        # decimate simulation output to have one wave per second, sampled at 30 fps
        print "    decimating..."
        x_arr_ds = x_arr[::decimation_step]
    
        # save simulation output
        filename = OUTPUT_PATH + "x_arr_ds__" + str(n_successful) + ".npy"
        print "saving simulation output to " + filename + "..."
        np.save(filename, np.array([[output_fps], x_arr_ds]))
    
        ## explicitly de-allocate memory (some of the arrays generated can be fairly large)
        print "de-allocating memory..."
        del(x_arr, x_arr_ds)
    
        ## increment count of successful simulations
        n_successful = n_successful + 1
    except (KeyboardInterrupt, SystemExit) :
        ## report keyboard shortcut and break loop
        print "Keyboard interrupt or system exit detected!"
        raise
    except :
        ## report error and increment error counter
        print "-- SIMULATION ERROR DETECTED -- "
        n_error = n_error + 1

print "Generated " + str(n) + " simulation outputs. There were " + str(n_error) + " aborted runs."



"""
Estimate the maximum Lyapunov characteristic exponent.
"""

print "estimating maximum Lyapunov characteristic exponent..."
print "    determining time axes..."
# LCE analysis parameters
dom_axial_freq = float(np.load(IC_PATH + "dom_axial_freq.npy")) # fundamental axial frequency
t_step = t_arr[1] - t_arr[0]            # time step
T = 1./dom_axial_freq                   # approximate orbital period
N_pb_per_T = 2                          # number of pullbacks per period
N_orbits = 1000                         # how many orbits to average over?
tol = 0.0006                            # simulation tolerance

pb_step = int((T/t_step)/N_pb_per_T)    # number of steps per pullback
n_pb = int(N_orbits*N_pb_per_T)         # number of pullbacks

print "    constructing mask to extract mechanical state..."
mech_mask = np.zeros(len(x0), dtype=np.bool)
mech_mask[3*N_seg:3*N_seg + 2*2*N] = True

print "    setting mechanical parameters..."
# no COM parameters
#[0.48]*(N_seg - 4) + [0.09]*4 + \
b_head = 0.05
b_body = 0.41
p0_no_COM = [-2.0,
      L0,
      -2.0] + \
     [b_body, b_head, b_head] + [b_body]*(N_seg - 4) + [b_head] + \
     [0.25] + \
     (1*np.array([0.067, 0.033, 1.67, 1.67, 1.67, 1.67, 1.67, 1.67, 0.67, 0.33])).tolist() + \
     [2, 1000] + \
     (1*0.2*np.array([0.35, 0.25] + [1]*(N_seg - 5) + [0.75, 0.5])).tolist() +\
     [2, 0.9] + \
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + \
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + \
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + \
     [0.01]
tol_no_COM = 0.0006

print "    performing MLCE estimation..."
# perform LCE analysis
lce_analysis = np.array(nm.analysis.lce_estimate(x0, f_f77, p0_no_COM,
    t_step=t_step/4., pb_step=500, n_pb=2000, n_ic_steps=15000, d0=10**-7,
    tol=0.0006, debug=False, n_pb_discard=10, dist=np.linalg.norm, log=np.log2,
    timescale=mech_timescale, mask=mech_mask))

LCE_filename = LCE_ANALYSIS_PATH + "lce_analysis.npy"
print "    saving result to " + LCE_filename
np.save(LCE_filename, lce_analysis)

#print "Plotting results..."
#lce_bpw = lce_analysis[0][0]
#lt_lce_estimates_bpw = lce_analysis[1]
#ft_lce_estimates_bpw = lce_analysis[2]
#
#plt.figure("Lyapunov characteristic exponent analysis")
#plt.clf()
#plt.subplot(211)
#plt.plot(lt_lce_estimates_bpw)
#plt.axhline(lce_bpw, c='b', lw=2)
#plt.ylabel("MLCE estimate")
#plt.grid(False)
#plt.subplot(212)
#plt.plot(ft_lce_estimates_bpw)
#plt.axhline(lce_bpw, c='b', lw=2)
#plt.xlabel("iteration")
#plt.ylabel("finite-time estimate")
#plt.grid(False)
#nm.util.hide_spines()
#plt.tight_layout()
