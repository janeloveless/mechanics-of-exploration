#! /usr/bin/env python2

import numpy as np
from sympy import S
import sympy as sy
import sympy.physics.mechanics as mech
import scipy as sp
import scipy.io
import matplotlib.pyplot as plt

import neuromech as nm
from neuromech.symbol import t

# TODO opening description

"""
Determine mode shapes and frequencies for the mechanical model, under the
assumption of small-amplitude motion and no friction.
"""

"""
Set some parameters. 

Some others will be defined later in the code, where it is more convenient.
"""

print "Setting parameters..."

# mechanical parameters
N = 12                                              # number of segments
m = sy.symbols("m", real=True, positive=True)       # segment mass
l = sy.symbols("l", real=True, positive=True)       # segment length
k_a = sy.symbols("k_a", real=True, positive=True)   # uniform axial stiffness parameter
k_t = sy.symbols("k_t", real=True, positive=True)   # uniform transverse stiffness parameter

# plotting parameters
output_dpi = 450
fontsize = 12
SAVE_PLOTS = True
SHOW_PLOTS = True
SAVE_ANIMATIONS = True
SHOW_ANIMATIONS = True
PLOT_PATH = "./data/output/1_modal_analysis/plots/"


"""
Build mechanical model.
"""

# NOTE that we set k_fluid=0. This is because we can impose the
# incompressibility constraint directly using kinematics for the case of small
# oscillations (see definition of new kinematics later in this script).
print "Building conservative mechanical model with uniform parameters..."
model = nm.model.SimplePlanarCrawler(N, 
        m=m,
        l=l,
        k_axial=k_a, 
        k_lateral=[k_t]*(N - 2), 
        k_fluid=0,
        n_axial=0, 
        n_lateral=[0]*(N - 2), 
        mu_f=[0]*N, 
        mu_b=[0]*N,
        b=[0]*(N - 1), 
        c=[0]*(N - 2))

print "Extracting model's original coordinate system and energies..."
qx = model.x[:2*N:2]            # original x coordinates
qy = model.x[1:2*N:2]           # original y coordinates
Px = model.x[2*N:4*N:2]         # original x momenta
Py = model.x[2*N + 1:4*N:2]     # original y momenta
       
H = model.H                     # original Hamiltonian
T = model.T                     # original kinetic energy
U = model.U                     # original potential energy
U_a = model.U_axial             # original axial potential energy
U_t = model.U_transverse        # original transverse potential energy


"""
Define new kinematics measured relative to equilibrium state.
"""

print "Defining new coordinates measured relative to equilibrium state..."

# x and y displacement from equilibrium
x = sy.Matrix(mech.dynamicsymbols("x_1:" + str(N + 1)))
y = sy.Matrix(mech.dynamicsymbols("y_1:" + str(N + 1)))

# conversion from equilibrium displacements to original coordinates
x_to_qx = x/sy.sqrt(m) + sy.Matrix([i*l for i in xrange(N)])
y_to_qy = y/sy.sqrt(m)

# strictly impose incompressibility constraint
x_to_qx = x_to_qx.subs(x[-1], x[0])
x = sy.Matrix(x[:-1])

# new configuration vector
w = sy.Matrix(np.concatenate([x, y]))

# momentum transformations (TODO prove that this transformation is canonical OR
# derive it explicitly in code)
px_to_Px = sy.Matrix(Px)*sy.sqrt(m)
py_to_Py = sy.Matrix(Py)*sy.sqrt(m)

# full coordinate transformation
z_old = sy.Matrix(np.concatenate([qx, qy, Px, Py]))
z_new_to_z_old = sy.Matrix(np.concatenate([x_to_qx, y_to_qy, px_to_Px, py_to_Py]))

# define equilibrium values for the new coordinates
x_eq = np.zeros(len(x))
y_eq = np.zeros(len(y))
w_eq = np.concatenate([x_eq, y_eq])


"""
Transform energetic quantities.
"""

print "Transforming energetic quantities to new coordinate system..."
T_xy = T.subs(zip(z_old, z_new_to_z_old)).simplify()
U_a_xy = U_a.subs(zip(z_old, z_new_to_z_old))
U_t_xy = U_t.subs(zip(z_old, z_new_to_z_old))


"""
Take quadratic approximation to the Hamiltonian.
"""

# find the axial stiffness matrix
print "Taking Hessian of axial potential energy (axial stiffness matrix)... (WARNING: SLOW!)"
K_a = sy.hessian(U_a_xy, w).subs(zip(w, w_eq))

# find the transverse stiffness matrix
print "Taking Hessian of transverse potential energy (trans. stiffness matrix)... (WARNING: SLOW!)"
K_t = sy.hessian(U_t_xy, w).subs(zip(w, w_eq))

# combine stiffness matrices
#print "Forming quadratic approximation to total potential energy..."
#K = K_a + K_t


"""
Find axial and transverse mode shapes from stiffness matrix.
"""

print "Dividing out scale factors in axial and transverse stiffness matrices..."
D2 = np.array((m/k_a)*K_a).astype(np.int)[:N - 1, :N - 1]
D4 = np.array((m*l**2/k_t)*K_t).astype(np.int)[N - 1:, N - 1:]

print "Axial stiffness matrix (check against paper definition of D2) :"
print D2

print "Transverse stiffness matrix (check against paper definition of D4) :"
print D4

print "Computing axial and transverse mode shapes and frequencies from stiffness matrices..."
# find axial mode shapes and frequencies
lam_a, v_a = np.linalg.eig(D2)          # find eigenvalues and eigenvectors of D2
v_a = v_a.T                             # mode shapes along first axis

# find transverse mode shapes and frequencies
lam_t, v_t = np.linalg.eig(D4)          # find eigenvalues and eigenvectors of D4
v_t = v_t.T                             # mode shapes along second axis

# sort modes by frequency
v_a = v_a[np.argsort(lam_a)]
lam_a = np.sort(lam_a)
v_t = v_t[np.argsort(lam_t)]
lam_t = np.sort(lam_t)


D4_szigeti__top_rows = np.array([[1, -2, 1] + [0]*67, [-2, 5, -4, 1] + [0]*66])
D4_szigeti__mid_rows = sp.linalg.circulant([1, -4, 6, -4, 1] + [0]*65)[4:]
D4_szigeti__bot_rows = np.array([[0]*66 + [1, -4, 5, -2], [0]*67 + [1, -2, 1]])
D4_szigeti = np.append(np.append(D4_szigeti__top_rows, 
                                 D4_szigeti__mid_rows, axis=0),
                                 D4_szigeti__bot_rows, axis=0)

lam_t_szigeti, v_t_szigeti = np.linalg.eig(D4_szigeti)
v_t_szigeti = v_t_szigeti.T
v_t_szigeti = v_t_szigeti[np.argsort(lam_t_szigeti)]
lam_t_szigeti = np.abs(np.sort(lam_t_szigeti))


"""
Transform our mode shapes into the postural frame used by Szigeti et al.
"""

print "Converting mode shapes into Szigeti postural frame..."
# x-axis for transverse mode shapes
ax_t = np.arange(12)

# rotate and scale the first four transverse mode shapes
# start by isolating the modes : they are listed after the first two modes,
# which correspond to overall rotations of the body
v_t1 = np.array([ax_t, -v_t[2]]).T
v_t2 = np.array([ax_t, -v_t[3]]).T
v_t3 = np.array([ax_t, v_t[4]]).T
v_t4 = np.array([ax_t, -v_t[5]]).T

# now move so that the tail point is at the origin
v_t1 = v_t1 - v_t1[0]
v_t2 = v_t2 - v_t2[0]
v_t3 = v_t3 - v_t3[0]
v_t4 = v_t4 - v_t4[0]

# then rotate so that the head point is on the x-axis
phi1 = -np.arctan2(v_t1[-1][1], v_t1[-1][0])
phi2 = -np.arctan2(v_t2[-1][1], v_t2[-1][0])
phi3 = -np.arctan2(v_t3[-1][1], v_t3[-1][0])
phi4 = -np.arctan2(v_t4[-1][1], v_t4[-1][0])
R1 = np.array([[np.cos(phi1), -np.sin(phi1)], [np.sin(phi1), np.cos(phi1)]])
R2 = np.array([[np.cos(phi2), -np.sin(phi2)], [np.sin(phi2), np.cos(phi2)]])
R3 = np.array([[np.cos(phi3), -np.sin(phi3)], [np.sin(phi3), np.cos(phi3)]])
R4 = np.array([[np.cos(phi4), -np.sin(phi4)], [np.sin(phi4), np.cos(phi4)]])
v_t1 = np.dot(v_t1, R1.T)
v_t2 = np.dot(v_t2, R2.T)
v_t3 = np.dot(v_t3, R3.T)
v_t4 = np.dot(v_t4, R4.T)

# then scale the x-axis so that the distance from head to tail is unity
v_t1[:, 0] = v_t1[:, 0]/v_t1[:, 0][-1]
v_t2[:, 0] = v_t2[:, 0]/v_t2[:, 0][-1]
v_t3[:, 0] = v_t3[:, 0]/v_t3[:, 0][-1]
v_t4[:, 0] = v_t4[:, 0]/v_t4[:, 0][-1]

# scale to unit maximum amplitude
v_t1[:, 1] = v_t1[:, 1] - np.mean(v_t1[:, 1])
v_t2[:, 1] = v_t2[:, 1] - np.mean(v_t2[:, 1])
v_t3[:, 1] = v_t3[:, 1] - np.mean(v_t3[:, 1])
v_t4[:, 1] = v_t4[:, 1] - np.mean(v_t4[:, 1])

v_t1[:, 1] = 0.5*v_t1[:, 1]/np.max(np.abs(v_t1[:, 1]))
v_t2[:, 1] = 0.5*v_t2[:, 1]/np.max(np.abs(v_t2[:, 1]))
v_t3[:, 1] = 0.5*v_t3[:, 1]/np.max(np.abs(v_t3[:, 1]))
v_t4[:, 1] = 0.5*v_t4[:, 1]/np.max(np.abs(v_t4[:, 1]))

# Add head displacement to axial modes. Head displacement and tail displacement
# are equal due to the presence of the incompressibility constraint.
print "Determining head displacement using strict incompressibility constraint..."

v_a1 = np.append(v_a[1], v_a[1][0])
v_a2 = np.append(v_a[2], v_a[2][0])
v_a3 = np.append(v_a[3], v_a[3][0])
v_a4 = np.append(v_a[4], v_a[4][0])


"""
Reconstruct Balazs Szigeti's PCA mode shapes.
"""

print "Reconstructing Szigeti eigenmaggot shapes..."

basis = sp.io.loadmat("./data/szigeti_modes/szigeti_modes.mat")["myBasis"][0][0][0].T

def reconstruct(eig_i) :
    rec_angles = basis[eig_i]
    rec_coords = np.zeros((len(rec_angles), 2))
    
    scale = 100./(len(rec_angles))
    
    for i in xrange(len(rec_angles) - 1) :
        rec_coords[i + 1, 0] = rec_coords[i, 0] + scale*np.cos(rec_angles[i])
        rec_coords[i + 1, 1] = rec_coords[i, 1] + scale*np.sin(rec_angles[i])

    return rec_coords
    
b_t1 = reconstruct(0)
b_t2 = reconstruct(1)
b_t3 = reconstruct(2)
b_t4 = reconstruct(3)

# scale to unit maximum amplitude
b_t1[:, 1] = b_t1[:, 1] - np.mean(b_t1[:, 1])
b_t2[:, 1] = b_t2[:, 1] - np.mean(b_t2[:, 1])
b_t3[:, 1] = b_t3[:, 1] - np.mean(b_t3[:, 1])
b_t4[:, 1] = b_t4[:, 1] - np.mean(b_t4[:, 1])

b_t1[:, 1] = 0.5*b_t1[:, 1]/np.max(np.abs(b_t1[:, 1]))
b_t2[:, 1] = 0.5*b_t2[:, 1]/np.max(np.abs(b_t2[:, 1]))
b_t3[:, 1] = 0.5*b_t3[:, 1]/np.max(np.abs(b_t3[:, 1]))
b_t4[:, 1] = 0.5*b_t4[:, 1]/np.max(np.abs(b_t4[:, 1]))
    

"""
Plot mode shapes.
"""

print "Plotting model mode shapes..."

plt.ioff()
fig = plt.figure("mode shapes", figsize=(7, 4.4))
plt.clf()
plot = fig.add_subplot(122) 
plt.title("axial modes", fontsize=fontsize)
scale=0.8
plt.plot(scale*v_a1/np.linalg.norm(v_a1) + 1, c='k', lw=2, marker="o")
plt.plot(scale*v_a2/np.linalg.norm(v_a2) + 2, c='k', lw=2, marker="o")
plt.plot(-scale*v_a3/np.linalg.norm(v_a3) + 3, c='k', lw=2, marker="o")
plt.plot(-scale*v_a4/np.linalg.norm(v_a4) + 4, c='k', lw=2, marker="o")
#plt.ylabel("axial mode number, $\\textbf{x}$")
plt.ylabel("mode no., axial displacement", fontsize=fontsize)
#plt.xlabel("segment boundary number")
plt.ylim(0, 5)
plt.yticks([1, 2, 3, 4])
plt.xlim(-0.5, 11.5)
plt.xticks([0, 11], ["tail", "head"])
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(121)
plt.title("transverse modes", fontsize=fontsize)
scale=0.8
plt.plot(scale*v_t1[:, 1]/np.linalg.norm(v_t1[:, 1]) + 1, lw=2, c='k', marker="o")
plt.plot(scale*v_t2[:, 1]/np.linalg.norm(v_t2[:, 1]) + 2, lw=2, c='k', marker="o")
plt.plot(scale*v_t3[:, 1]/np.linalg.norm(v_t3[:, 1]) + 3, lw=2, c='k', marker="o")
plt.plot(scale*v_t4[:, 1]/np.linalg.norm(v_t4[:, 1]) + 4, lw=2, c='k', marker="o")
#plt.ylabel("transverse mode number, $\\textbf{y}$")
plt.ylabel("mode no., transverse displacement", fontsize=fontsize)
#plt.xlabel("segment boundary number")
plt.ylim(0, 5)
plt.yticks([1, 2, 3, 4])
plt.xlim(-0.5, 11.5)
plt.xticks([0, 11], ["tail", "head"])
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.tight_layout()
plt.grid(False)


print "Plotting Szigeti eigenmaggot shapes..."

head_disp = 0.107
plot = fig.add_subplot(121)
plt.plot(b_t1[:, 0]*head_disp, scale*b_t1[:, 1]/np.linalg.norm(v_t1[:, 1]) + 1,
        lw=3, c='b', alpha=0.5)
plt.plot(b_t2[:, 0]*head_disp, scale*b_t2[:, 1]/np.linalg.norm(v_t2[:, 1]) + 2,
        lw=3, c='b', alpha=0.5)
plt.plot(b_t3[:, 0]*head_disp, -scale*b_t3[:, 1]/np.linalg.norm(v_t3[:, 1]) +
        3, lw=3, c='b', alpha=0.5)
plt.plot(b_t4[:, 0]*head_disp, scale*b_t4[:, 1]/np.linalg.norm(v_t4[:, 1]) + 4,
        lw=3, c='b', alpha=0.5)
plt.tight_layout()
nm.util.hide_spines()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "mode_shapes.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


"""
Plot axial and transverse spectra.
"""

print "Plotting model frequencies..."

plt.ioff()
fig = plt.figure("modal frequencies", figsize=(7, 2.2))
plt.clf()
plot = fig.add_subplot(122)
plt.title("axial frequencies", fontsize=fontsize)
plt.scatter(np.arange(len(lam_a) - 1) + 1, lam_a[1:]/np.max(lam_a), c='k', s=30)
plt.plot([-1, 4.5, 4.5], [0.45, 0.45, -1], c='k', lw=2, alpha=0.2)
plt.xticks(np.arange(len(lam_a) - 1) + 1)
plt.xlim(0.5, N - 2 + 0.5)
plt.ylim(-0.1, 1.1)
plt.grid(False)
plt.xlabel("mode number", fontsize=fontsize)
plt.yticks([0, 1])
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(121)
plt.title("transverse frequencies", fontsize=fontsize)
plt.scatter(np.arange(len(lam_t) - 2) + 1, lam_t[2:]/np.max(lam_t), c='k', s=30)
plt.plot([-1, 4.5, 4.5], [0.45, 0.45, -1], c='k', lw=2, alpha=0.2)
plt.xticks(np.arange(len(lam_t) - 2) + 1)
plt.xlim(0.5, N - 2 + 0.5)
plt.ylim(-0.1, 1.1)
plt.yticks([0, 1])
plt.xlabel("mode number", fontsize=fontsize)
plt.ylabel("$\omega$/max($\omega$)", fontsize=fontsize)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.tight_layout()
nm.util.hide_spines()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "mode_frequencies.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


"""
Plot a forward and backward wave using the lowest frequency axial modes.
"""

print "Generating forward / backward wave motion..."

n_waves = 5
md1 = v_a1/np.max(v_a1)
md2 = v_a2/np.max(v_a2)

t_arr = np.linspace(0, n_waves*2*np.pi, 1000)
co1_fw = np.cos(t_arr)
co2_fw = np.sin(t_arr)
co1_bk = np.sin(t_arr)
co2_bk = np.cos(t_arr)

fw_wave = np.array([md1_i*co1_fw + md2_i*co2_fw for md1_i, md2_i in zip(md1, md2)])
fw_wave = (fw_wave.T/np.max(fw_wave, axis=1)).T
bk_wave = np.array([md1_i*co1_bk + md2_i*co2_bk for md1_i, md2_i in zip(md1, md2)])
bk_wave = (bk_wave.T/np.max(bk_wave, axis=1)).T

print "Plotting forward / backward wave motion..."

V = 3   # COM velocity (larva-like waves)
V = 0   # COM velocity (no overall motion)
md_amp = 0.6  # modal amplitude
plt.ioff()
fig = plt.figure("axial travelling waves", figsize=(3.3, 7))
plt.clf()
plot = fig.add_subplot(211)
plt.title("forward axial wave", fontsize=fontsize)
plt.plot(t_arr/(2*np.pi), (V*t_arr/(2*np.pi) + (md_amp*fw_wave.T +
    np.arange(12)).T).T, lw=2)
plt.ylabel("axial displacement (segs)", fontsize=fontsize)
plt.ylim(-1, 12)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)

plot = fig.add_subplot(212)
plt.title("backward axial wave")
plt.plot(t_arr/(2*np.pi), (-V*t_arr/(2*np.pi) + (md_amp*bk_wave.T +
    np.arange(12)).T).T, lw=2)
plt.ylabel("axial displacement (segs)", fontsize=fontsize)
plt.ylim(-1, 12)
plt.xlabel("time (s)", fontsize=fontsize)
plt.grid(False)
plot.tick_params(axis="both", which="major", labelsize=fontsize)
plt.tight_layout()
nm.util.hide_spines()
if SAVE_PLOTS : plt.savefig(PLOT_PATH + "travelling_waves.png", dpi=output_dpi)
if not SHOW_PLOTS : plt.close()
plt.ion()
plt.show()


"""
Animate axial modes.
"""

print "Animating low-frequency axial motion..."

plt.ioff()
import matplotlib.animation as animation
# Set up formatting for the movie files
Writer = animation.writers['imagemagick']
writer = Writer(fps=30, bitrate=1800)

fig = plt.figure("low frequency axial modes animation", figsize=(7, 7))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.5, 11.5), ylim=(-3, 1))

line_mode1, = ax.plot([], [], 'o-', lw=2, alpha=0.2, c='b')
line_mode2, = ax.plot([], [], 'o-', lw=2, alpha=0.2, c='r')
line_travelling_soln, = ax.plot([], [], 'o-', lw=2, c='k')
line_travelling_soln_axial1, = ax.plot([], [], '|-', lw=2, c='k', markersize=20)
line_travelling_soln_axial2, = ax.plot([], [], '|-', lw=2, c='k', markersize=20)
line_travelling_soln_axial3, = ax.plot([], [], "o", lw=2, c='k', markersize=8)

plt.tight_layout()
plt.grid(False)
plt.xticks([])
plt.yticks([])
#nm.util.hide_spines()

def init_axial():
    line_mode1.set_data([], [])
    line_mode2.set_data([], [])
    line_travelling_soln.set_data([], [])
#    line_travelling_soln_axial1.set_data([], [])
    line_travelling_soln_axial2.set_data([], [])
    line_travelling_soln_axial3.set_data([], [])
    return line_mode1, line_mode2, line_travelling_soln, line_travelling_soln_axial1, line_travelling_soln_axial2, line_travelling_soln_axial3

n_waves=10
t_arr = np.linspace(0, n_waves*2*np.pi, 1000)
scale = 0.8

def animate_axial(i):
    mode1_i = scale*v_a1*np.cos(t_arr[i])
    mode2_i = scale*v_a2*np.sin(t_arr[i])
    travelling_wave_i = mode1_i + mode2_i

    line_mode1.set_data(np.arange(12), mode1_i)
    line_mode2.set_data(np.arange(12), mode2_i)
    line_travelling_soln.set_data(np.arange(12), travelling_wave_i)
#    line_travelling_soln_axial1.set_data(np.arange(12) + travelling_wave_i, -1*np.ones(N))
    line_travelling_soln_axial2.set_data(0.5*np.arange(12) +
            0.9*travelling_wave_i + 0.25*t_arr[i] - 5.5, -2*np.ones(N))
    line_travelling_soln_axial3.set_data(0.5*np.arange(12) +
            0.9*travelling_wave_i + 0.25*t_arr[i] - 5.5, -2*np.ones(N))
    return line_mode1, line_mode2, line_travelling_soln, line_travelling_soln_axial1, line_travelling_soln_axial2, line_travelling_soln_axial3

ani = animation.FuncAnimation(fig, animate_axial, np.arange(1, len(t_arr), 1),
    interval=25, blit=True, init_func=init_axial)

if SAVE_ANIMATIONS : ani.save(PLOT_PATH + 'axial_animation.gif', writer=writer)
if not SHOW_ANIMATIONS : plt.close()
plt.ion()
plt.show()



"""
Animate transverse modes.
"""

print "Animating transverse motion..."

plt.ioff()
import matplotlib.animation as animation
# Set up formatting for the movie files
Writer = animation.writers['imagemagick']
writer = Writer(fps=30, bitrate=1800)

fig = plt.figure("transverse modes animation", figsize=(7, 7))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.5, 11.5), ylim=(-1, 4))

line_mode1, = ax.plot([], [], 'o-', lw=2, alpha=1.0, c='k')
line_mode2, = ax.plot([], [], 'o-', lw=2, alpha=1.0, c='k')
line_mode3, = ax.plot([], [], 'o-', lw=2, alpha=1.0, c='k')
line_mode4, = ax.plot([], [], 'o-', lw=2, alpha=1.0, c='k')
line_combined, = ax.plot([], [], 'o-', lw=2, alpha=0.2, c='k')

plt.tight_layout()
plt.grid(False)
plt.xticks([])
plt.yticks([])
#nm.util.hide_spines()

def init_transverse():
    line_mode1.set_data([], [])
    line_mode2.set_data([], [])
    line_mode3.set_data([], [])
    line_mode4.set_data([], [])
    line_combined.set_data([], [])
    return line_mode1, line_mode2, line_mode3, line_mode4, line_combined

n_waves=10
t_arr = np.linspace(0, n_waves*2*np.pi, 1000)
scale = 0.8

freq_factor = 0.2
freq1 = np.sqrt(lam_t)[2]
freq2 = (np.sqrt(lam_t)[3]/freq1)*freq_factor
freq3 = (np.sqrt(lam_t)[4]/freq1)*freq_factor
freq4 = (np.sqrt(lam_t)[5]/freq1)*freq_factor
freq1 = freq_factor
def animate_transverse(i):
    mode1_i = scale*v_t1[:, 1]*np.cos(t_arr[i]*freq1)
    mode2_i = scale*v_t2[:, 1]*np.cos(t_arr[i]*freq2)
    mode3_i = scale*v_t3[:, 1]*np.cos(t_arr[i]*freq3)
    mode4_i = scale*v_t4[:, 1]*np.cos(t_arr[i]*freq4)

    line_mode1.set_data(np.arange(12), mode1_i + 0)
    line_mode2.set_data(np.arange(12), mode2_i + 1)
    line_mode3.set_data(np.arange(12), mode3_i + 2)
    line_mode4.set_data(np.arange(12), mode4_i + 3)
    return line_mode1, line_mode2, line_mode3, line_mode4, line_combined

ani = animation.FuncAnimation(fig, animate_transverse, np.arange(1, len(t_arr), 1),
    interval=25, blit=True, init_func=init_transverse)

if SAVE_ANIMATIONS : ani.save(PLOT_PATH + 'transverse_animation.gif', writer=writer)
if not SHOW_ANIMATIONS : plt.close()
plt.ion()
plt.show()

