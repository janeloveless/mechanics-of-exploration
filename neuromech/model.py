#! /usr/bin/env python

import sympy as sy
import sympy.physics.mechanics as mech
import numpy as np
import scipy as sp

import util
from symbol import t, V, I


# TODO simplify SimplePlanarCrawler
# TODO rename SimplePlanarCrawler

# TODO move definition of head mechanical model into this submodule

"""
Generic model classes.
"""

class Model(object) : 
    def __init__(self, parameters=None) :
        self.parameters=parameters

    def subs(self, subs_list) :
        # all subclasses should be able to take a list of symbolic
        # substitutions and execute these for all symbolic expressions
        # belonging to the class
        raise NotImplementedError


class DynamicalModel(Model) :
    def __init__(self, x=None, parameters=None, f=None, jacobian=None,
            f_num=None, jacobian_num=None, FORTRAN_f=None,
            FORTRAN_jacobian=None) : 
        self.x = x                                   # state variables
        self.f = f                                   # state evolution rule
        self._jacobian = jacobian                    # jacobian of state evolution rule
        self._f_num = f_num                          # callable state evolution rule
        self._jacobian_num = jacobian_num            # callable jacobian function
        self._FORTRAN_f = FORTRAN_f                  # FORTRAN source for state evolution rule
        self._FORTRAN_jacobian = FORTRAN_jacobian    # FORTRAN source for jacobian function

    @property
    def parameters(self) :
        params = []
        for param in self.f.free_symbols.difference(self.x).difference({t}) :
            if type(param) != sy.stats.rv.RandomSymbol :
                params.append(param)
        params = np.array(params)
        sort_i = np.argsort(params.astype(np.str))
        params = params[sort_i].tolist()
        return params


    def jacobian(self) :
        # TODO parallelise -- this is SLOW but must be done in SymPy;
        # it should be possible to compute each entry in the Jacobian matrix
        # independently
        self._jacobian = sy.Matrix(self.f).jacobian(self.x)
        return self._jacobian

    def f_num(self) :
        f_lambdified = sy.lambdify([t] + self.x, self.f)
        self._f_num = lambda x, t : np.array(f_lambdified(t, *x), dtype=np.float).flatten() 
        return self._f_num

    def jacobian_num(self, new_jac=False) :
        if self._jacobian is None or new_jac is True :
            self.jacobian()

        jac_lambdified = sy.lambdify([t] + self.x, self._jacobian)
        self._jacobian_num = lambda x, t : np.array(jac_lambdified(t, *x)) 
        return self._jacobian_num

    def FORTRAN_f(self, verbose=False) :
        self._FORTRAN_f = util.FORTRAN_f(self.x, self.f, self.parameters,
                verbose)
        return self._FORTRAN_f

    def FORTRAN_jacobian(self, new_jac=False) :
        if self._jacobian is None or new_jac is True :
            self.jacobian()

        self._FORTRAN_jacobian = util.FORTRAN_jacobian(self.x, self._jacobian, self.parameters)
        return self._FORTRAN_jacobian


"""
Mechanical modelling.
"""

def coulomb_friction_function(p, mu_f, mu_b) :
    return sy.Piecewise((-mu_f, p > 0), (mu_b, p < 0), (0, True))


def derive_Hamiltons_equations(H, q, p, Q=None) :
    """
    Derive equations of motion for a Hamiltonian system.
    
    Arguments
    ---------
        H    : Hamiltonian for the system
        q    : vector of generalised coordinates
        p    : vector of generalised momenta
        Q    : vector of generalised forces
        
        
    Returns
    -------
        x'   : dynamical rule of evolution for the system. Note that x is the full 
               state vector for the system, x = [q | p].T
    """
    if Q is None : Q = np.zeros(len(q))
    q_dot = [sy.diff(H, p_i) for p_i in p]
    p_dot = [-sy.diff(H, q[i]) + Q[i] for i in xrange(len(q))]
    return sy.Matrix(q_dot + p_dot)


class MechanicalSystem(DynamicalModel) : 
    def __init__(self, q, p, H, Q=None, u=None, timescale=1.) :
        """
        Construct the equations of motion for a mechanical system, given a
        vector of generalised coordinates q, vector of conjugate momenta p,
        Hamiltonian function H, Rayleigh dissipation function R, a vector of
        generalised forces Q, and a vector of control inputs u. Often Q will be
        a symbolic function of u.
        """
        self.q = q
        self.p = p
        self.H = H
        self.Q = Q
        self.u = u

        self.x = list(q) + list(p)
        self.f = derive_Hamiltons_equations(H, q, p, Q=Q)*timescale


    def H_num(self) :
        H_lambdified = sy.lambdify([t] + self.x, self.H)
        self._H_num = lambda x, t : np.array(H_lambdified(t, *x),
                dtype=np.float).flatten()
        return self._H_num


class ConservativeHead(MechanicalSystem) :
    def __init__(self, lam=sy.symbols("lambda"), eps=sy.symbols("epsilon"),
            **kwargs) :
        # define coordinates and momenta 
        q = mech.dynamicsymbols("q")                        # axial strain
        phi = mech.dynamicsymbols("phi")                    # bending angle
        p_q = mech.dynamicsymbols("p_q")                    # axial momentum
        p_phi = mech.dynamicsymbols("p_phi")                # bending momentum

        # define energetic quantities
        T = sy.S("1/2")*p_q**2 + \
            sy.S("1/2")*(1/((1 + eps*q)**2))*(p_phi**2)     # kinetic energy
        U_a = sy.S("1/2")*q**2                              # axial potential
        U_t = sy.S("1/2")*lam**2*phi**2                     # transverse potential
        U = U_a + U_t                                       # total potential
        H = T + U                                           # Hamiltonian

        super(ConservativeHead, self).__init__([q, phi], [p_q, p_phi], H, **kwargs)


class NondimensionalHarmonicCrawler(MechanicalSystem) :
    def __init__(self, N, w0=sy.symbols("omega_0"), Z=sy.symbols("zeta"),
            mu_f=sy.symbols("mu_f"), mu_b=sy.symbols("mu_b"),
            b=sy.symbols("b"), **kwargs) :

        # construct position, momentum, and control vectors
        q = sy.Matrix([mech.dynamicsymbols("q"+str(i + 1)) for i in xrange(N)])
        p = sy.Matrix([mech.dynamicsymbols("p"+str(i + 1)) for i in xrange(N)])
        u = sy.Matrix([mech.dynamicsymbols("u"+str(i + 1)) for i in xrange(N)])
        
        # construct some useful matrices; scale parameters
        if N > 1 :
            Z = sy.S("1/4")*Z
            w0 = sy.S("1/2")*w0
            D1 = -sy.Matrix(sp.linalg.circulant([-1, 1] + [0]*(N - 2)))
        else :
            D1 = sy.Matrix([1])

        D2 = D1.T*D1

        # construct the stiffness matrix
        K = (w0**2)*D2

        # form Hamiltonian function using matrix math, but then write products
        # explicitly (this is useful later as it simplifies differentiation and
        # some other SymPy functions)
        H = sy.S("1/2")*(p.T*p + q.T*K*q)
        H = H.as_immutable().as_explicit()[0]
        
        # generalised forces due to control input
        Q_u = b*D1*u
        
        # generalised forces due to viscous friction 
        Q_n = -2*Z*w0*D2*p
        
        # generalised forces due to dry friction
        Q_F = sy.Matrix([coulomb_friction_function(p_i, mu_f, mu_b) for p_i in p])
        
        # combine generalised forces
        Q = Q_u + Q_n + Q_F

        # call superconstructor
        super(NondimensionalHarmonicCrawler, self).__init__(q, p, H, Q, u, **kwargs)

        # form lists of state and control variables according to body segment
        self.seg_x = [self.x[i::len(self.x)/2] for i in xrange(len(self.x)/2)]
        self.seg_u = self.u


class SimplePlanarCrawler(MechanicalSystem) :
    def __init__(self, N=12, 
        m=sy.symbols("m"),                                  # segment mass
        l=sy.symbols("l"),                                  # equilibrium segment length
        L=sy.symbols("L"),                                  # equilibrium body length
        k_axial=sy.symbols("k_axial"),                      # axial stiffness
        k_lateral=sy.symbols("k_lateral_2:" + str(12)),     # transverse stiffness
        k_fluid=sy.symbols("k_fluid"),                      # fluid stiffness
        n_axial=sy.symbols("eta_axial"),                    # axial viscosity
        n_lateral=sy.symbols("eta_lateral_2:" + str(12)),   # transverse viscosity
        mu_f=sy.symbols("mu_f_1:" + str(13)),               # forward dry friction coefficient
        mu_b=sy.symbols("mu_b_1:" + str(13)),               # backward dry friction coefficient 
        mu_p=sy.symbols("mu_p_1:" + str(13)),               # dry friction power (focus)
        b=sy.symbols("b_1:" + str(12)),                     # axial control gain
        c=sy.symbols("c_2:" + str(12))) :                   # transverse control gain
        """
        """
        # TODO add docstring


        #################################################################
        # define useful functions 
        #################################################################

        norm = lambda x : sy.sqrt(np.dot(x, x))


        #################################################################
        # define kinematic quantities
        #################################################################

        t = sy.symbols("t")
        
        # generalised coordinates, giving displacement of each mass relative to lab frame
        qx = mech.dynamicsymbols("q_1:" + str(N + 1) + "_x")
        qy = mech.dynamicsymbols("q_1:" + str(N + 1) + "_y")
        q_vecs = np.array([qx, qy]).T
        q = q_vecs.flatten()
        
        # axial vectors pointing along the body axis
        q_diffs = np.diff(q_vecs, axis=0)
        
        # conjugate momenta, giving translational momentum of each mass relative to lab frame
        px = mech.dynamicsymbols("p_1:" + str(N + 1) + "_x")
        py = mech.dynamicsymbols("p_1:" + str(N + 1) + "_y")
        p_vecs = np.array([px, py]).T
        p = p_vecs.flatten()
        
        # coordinate transformation from q's to phi's
        phi_to_q = []
        for i in xrange(1, N - 1) :
            rd1 = q_diffs[i - 1]
            rd2 = q_diffs[i]
        
            angle = sy.atan2(rd1[0]*rd2[1] - rd2[0]*rd1[1], 
                             rd1[0]*rd2[0] + rd1[1]*rd2[1]);
            phi_to_q.append(angle)
        Dphi_to_Dq = [sy.diff(phi_to_q__i, t) for phi_to_q__i in phi_to_q]
 
        # rs in terms of qs
        r_to_q = [norm(q_diff) for q_diff in q_diffs]
        Dr_to_Dq = [sy.diff(r_to_q__i, t) for r_to_q__i in r_to_q]
        
        # generalised velocities
        Dqx = mech.dynamicsymbols("q_1:" + str(N + 1) + "_x", 1)
        Dqy = mech.dynamicsymbols("q_1:" + str(N + 1) + "_y", 1)
        Dq_vecs = np.array([Dqx, Dqy]).T
        Dq = Dq_vecs.flatten()
        
        # momenta in terms of velocities
        Dq_to_p = p*m   # TODO double-check this
                        # TODO derive this from Hamiltonian using Hamilton's
                        # equation

    
        #################################################################
        # define energetic quantities
        #################################################################

        # kinetic energy
        T = (1/(2*m))*np.sum(p**2)
        
        # axial (stretch) elastic energy
        U_axial = sy.S("1/2")*k_axial*np.sum((np.array(r_to_q) - l)**2)
        
        # lateral (bending) elastic energy
        U_lateral = 0
        for i in xrange(1, N - 1) :
            U_lateral += k_lateral[i - 1]*sy.acos(np.dot(q_diffs[i], q_diffs[i - 1])/ \
                                                  (norm(q_diffs[i])*norm(q_diffs[i - 1])))
        U_lateral = sy.S("1/2")*U_lateral
        U_lateral = sy.S("1/2")*np.dot(k_lateral, (np.array(phi_to_q))**2)
        
        # fluid elastic energy
        U_fluid = sy.S("1/2")*k_fluid*(np.sum(r_to_q) - L)**2
        
        # total potential energy
        U = U_axial + U_lateral + U_fluid

        # axial dissipation function (viscosity)
        R_axial = sy.S("1/2")*n_axial*np.sum(np.array(Dr_to_Dq)**2)
        
        # lateral dissipation function (viscosity)
        R_lateral = sy.S("1/2")*np.dot(n_lateral, np.array(Dphi_to_Dq)**2)

        # axial dissipation function (control)
        #b = sy.symbols("b_1:" + str(N))             # axial gains
        #u = mech.dynamicsymbols("u_1:" + str(N))    # axial control variables
        #R_u = S("1/2")*np.sum([-b_i*u_i*Dq_i for b_i, u_i, Dq_i in zip(b, u, Dr_to_Dq)])

        # lateral dissipation function (control)
        v = mech.dynamicsymbols("v_2:" + str(N))    # lateral control variables
        R_v = sy.S("1/2")*np.sum([c_i*v_i*Dphi_i for c_i, v_i, Dphi_i in zip(c, v, Dphi_to_Dq)])

        # Hamiltonian, H, describing total energy and 
        # Rayleigh dissipation function, R, describing total power losses
        H = T + U
        R = R_axial + R_lateral + R_v

        # store energetic quantities in object variable
        self.H = H                      # Hamiltonian
        self.T = T                      # kinetic energy
        self.U = U                      # potential energy
        self.U_axial = U_axial          # axial potential energy
        self.U_transverse = U_lateral   # transverse potential energy
        self.U_fluid = U_fluid          # fluid potential energy
        self.R = R                      # Rayleigh dissipation function
        self.R_axial = R_axial          # axial dissipation function
        self.R_transverse = R_lateral   # transverse dissipation function
        self.R_v = R_v                  # transverse control dissipation function


        #################################################################
        # derive / construct generalised forces
        #################################################################

        # derive dissipative forces in terms of momentum variables
        Q_R = []
        for Dqi in Dq :
            print "Computing dissipative forces associated with " + str(Dqi) + "..."
            Q_R.append(-sy.diff(R, Dqi).subs(zip(Dq, Dq_to_p)))


        # derive forces due to control input
        u = mech.dynamicsymbols("u_1:" + str(N))
        Q_u = np.sum(np.array([-b_i*u_i*np.array([sy.diff(r_to_q_i, q_i) for q_i in q]) 
                    for b_i, u_i, r_to_q_i in zip(b, u, r_to_q)]).T, axis=1)
        
        # derive forces due to dry friction
        R = lambda theta : sy.Matrix([[sy.cos(theta), -sy.sin(theta)], [sy.sin(theta), sy.cos(theta)]])
        
        # find unit linear momentum vectors
        p_vecs_unit = [p_vec/sy.sqrt(np.dot(p_vec, p_vec)) for p_vec in p_vecs]
        
        # find unit vectors pointing along "spine"
        spine_vecs_unit = [q_diff_vec/sy.sqrt(np.dot(q_diff_vec, q_diff_vec)) for q_diff_vec in q_diffs]
        spine_vecs_unit += [spine_vecs_unit[-1]]
        spine_vecs_unit = [sy.Matrix(spine_vec) for spine_vec in spine_vecs_unit]
        
        # find rotation matrices to transform from spine vectors to segment orientation (n) vectors
        n_R_matrices = [R(0)] + [R(phi_i) for phi_i in phi_to_q] + [R(0)]
        
        # transform to n vectors
        n_vecs = [n_R*spine_vec for n_R, spine_vec in zip(n_R_matrices, spine_vecs_unit)]
        
        # find angle of momentum vector relative to n vector
        p_angles = [sy.acos(sy.Matrix(p_unit).T*n_vec) for p_unit, n_vec in zip(p_vecs_unit, n_vecs)]
            
        # use angle to find magnitude of friction force
        # NOTE this block tends to fail with a NotImplementedError in sympy
        for i in xrange(len(p_angles)) :
            try :
                sy.cos(p_angles[i])
            except :
                print "failure " + str(i)

        _cos = [sy.cos(p_angles[i])[0] for i in xrange(len(p_angles))]
        Q_mags = [mu_f[i] + (mu_b[i] - mu_f[i])*((1 - _cos[i])/2.)**mu_p[i] for i in xrange(len(p_angles))]

        # compute friction force
        Q_friction = [-Q_mag*p_unit for Q_mag, p_unit in zip(Q_mags, p_vecs_unit)]
        Q_friction = np.array(Q_friction).flatten()

        Q = np.array(Q_u) + np.array(Q_R) + np.array(Q_friction)

        # use superconstructor to derive equations of motion
        super(SimplePlanarCrawler, self).__init__(q, p, H, Q=Q, u=u + v)


class ConservativeSimplePlanarCrawler(MechanicalSystem) :
    def __init__(self, N=12) :
        #################################################################
        # define parameters
        #################################################################

        m = sy.symbols("m")                                 # mass
        l = sy.symbols("l")                                 # equilibrium segment length
        L = sy.symbols("L")                                 # equilibrium body length
    
        k_axial = sy.symbols("k_axial")                     # axial stiffness
        k_lateral = sy.symbols("k_lateral_2:" + str(N))     # bending stiffness
        k_fluid = sy.symbols("k_fluid")                     # fluid stiffness
        

        #################################################################
        # define useful functions
        #################################################################

        norm = lambda x : sy.sqrt(np.dot(x, x))


        #################################################################
        # define kinematic quantities
        #################################################################

        # generalised coordinates, giving displacement of each mass relative to lab frame
        qx = mech.dynamicsymbols("q_1:" + str(N + 1) + "_x")
        qy = mech.dynamicsymbols("q_1:" + str(N + 1) + "_y")
        q_vecs = np.array([qx, qy]).T
        q = q_vecs.flatten()
        
        # axial vectors pointing along the body axis
        q_diffs = np.diff(q_vecs, axis=0)
        
        # conjugate momenta, giving translational momentum of each mass relative to lab frame
        px = mech.dynamicsymbols("p_1:" + str(N + 1) + "_x")
        py = mech.dynamicsymbols("p_1:" + str(N + 1) + "_y")
        p_vecs = np.array([px, py]).T
        p = p_vecs.flatten()
        
        # coordinate transformation from q's to phi's
        phi_to_q = []
        for i in xrange(1, N - 1) :
            rd1 = q_diffs[i - 1]
            rd2 = q_diffs[i]
        
            angle = sy.atan2(rd1[0]*rd2[1] - rd2[0]*rd1[1], 
                             rd1[0]*rd2[0] + rd1[1]*rd2[1]);
            phi_to_q.append(angle)
 
        
        # rs in terms of qs
        r_to_q = [norm(q_diff) for q_diff in q_diffs]
        
    
        #################################################################
        # define energetic quantities
        #################################################################

        # kinetic energy
        T = (1/(2*m))*np.sum(p**2)
        
        # axial (stretch) elastic energy
        U_axial = sy.S("1/2")*k_axial*np.sum((np.array(r_to_q) - l)**2)
        
        # lateral (bending) elastic energy
        U_lateral = 0
        for i in xrange(1, N - 1) :
            U_lateral += k_lateral[i - 1]*sy.acos(np.dot(q_diffs[i], q_diffs[i - 1])/ \
                                                  (norm(q_diffs[i])*norm(q_diffs[i - 1])))
        U_lateral = sy.S("1/2")*U_lateral
        U_lateral = sy.S("1/2")*np.dot(k_lateral, (np.array(phi_to_q))**2)
        
        # fluid elastic energy
        U_fluid = sy.S("1/2")*k_fluid*(np.sum(r_to_q) - L)**2
        
        # total potential energy
        U = U_axial + U_lateral + U_fluid
        
        # Hamiltonian (total energy)
        H = T + U

        # use superconstructor to derive equations of motion
        super(ConservativeSimplePlanarCrawler, self).__init__(q, p, H)


"""
Neural modelling.
"""

class DynamicalNeuron(DynamicalModel) : 
    def __init__(self, *args, **kwargs) :
        super(DynamicalNeuron, self).__init__(*args, **kwargs)


class DynamicalBinaryNeuron(DynamicalNeuron) : 
    def __init__(self, w=[sy.symbols("w")], u=[I], theta=sy.symbols("theta"),
            x=V, k=sy.symbols("k")) :
        self.x = [x]            # state vector
        self.w = w              # weight vector
        self.u = u              # input vector
        self.theta = theta      # threshold
        self.k = k              # rate constant for state transition

    @property
    def f(self) :
        x = self.x[0]
        w = self.w
        u = self.u
        k = self.k
        return sy.Matrix([sy.Piecewise((k - k*x, np.dot(w, u) > self.theta), 
                                       (-k*x, True))])

"""
Set up the neuromuscular system, consisting of three cell types : sensory
neuron (SN), inhibitory interneuron (IN), and motor neuron (MN). This model
includes no muscle fibres; the MNs directly produce forces! There is one cell
of each type within each segment. They are connected to each other and to the
mechanical system as follows :
        
    mechanics -> SN
    SN -> MN
        -> IN
    IN -> MN (neighbouring segment some distance away)
        -> IN (neighbouring segment some distance away)
    MN -> mechanics
        
Note that the INs form a "mutual inhibition" network.
"""
class MechanicalFeedbackAndMutualInhibition(DynamicalModel) :
    def __init__(self, N_seg, 
        # TODO provide symbolic SN_u, SN_ws!
        SN_u,                                   # vector of sensory neuron inputs
        SN_ws,                                  # matrix of sensory neuron input weights
        k=1,                                    # binary neuron switching rate
        SN_thresh=sy.symbols("theta_SN"),
        IN_SN_w=1,                              # sensory neuron -> inhibitory interneuron weight 
        #IN_IN_w=-2,                            # inh interneuron -> inhibitory interneuron weight 
        IN_IN_w=sy.symbols("IN_IN_w"),
        IN_thresh=0.5,                          # IN threshold for activation
        MN_SN_w=1,                              # sensory neuron -> motor neuron weight
        #MN_IN_w=-2,                            # inhibitory interneuron -> motor neuron weight 
        MN_IN_w=sy.symbols("MN_IN_w"),
        MN_thresh=0.5) :                        # MN threshold before activation
        
        # state variables for each neuron population
        V_SNs = [sy.symbols("V_SN_" + str(i + 1)) for i in xrange(N_seg)]
        V_INs = [sy.symbols("V_IN_" + str(i + 1)) for i in xrange(N_seg)]
        V_MNs = [sy.symbols("V_MN_" + str(i + 1)) for i in xrange(N_seg)]
        
        # construct sensory neuron population
        print "Constructing sensory neuron population..."
        SNs = [DynamicalBinaryNeuron(w, SN_u, SN_thresh, r, k) for w, r in zip(SN_ws, V_SNs)]

        # set inhibitory interneuron inputs :
        #   SN -> IN within the same segment
        #   IN -> IN across non-adjacent segments
        print "Setting inhibitory interneuron input weights..."
        IN_u = V_SNs + V_INs
        IN_SN_ws = (IN_SN_w*np.eye(N_seg)).tolist()
        IN_IN_adj = sp.linalg.circulant([0, 0] + [1]*(N_seg - 3) + [0])  
        IN_IN_ws = (IN_IN_w*IN_IN_adj).tolist()                             
        IN_ws = [SN_w + IN_w for SN_w, IN_w in zip(IN_SN_ws, IN_IN_ws)] 
        
        # construct inhibitory interneuron population
        print "Constructing inhibitory interneuron population..."
        INs = [DynamicalBinaryNeuron(w, IN_u, IN_thresh, r, k) for w, r in zip(IN_ws, V_INs)]
        
        # set motor neuron inputs :
        #   SN -> MN within the same segment
        #   IN -> MN across non-adjacent segments
        print "Setting motor neuron input weights..."
        MN_u = V_SNs + V_INs
        MN_SN_ws = (MN_SN_w*np.eye(N_seg)).tolist()  
        MN_IN_adj = IN_IN_adj 
        MN_IN_ws = (MN_IN_w*MN_IN_adj).tolist()
        MN_ws = [SN_w + IN_w for SN_w, IN_w in zip(MN_SN_ws, MN_IN_ws)]  
        
        print "Constructing motor neuron population..."
        MNs = [DynamicalBinaryNeuron(w, MN_u, MN_thresh, r, k) for w, r in zip(MN_ws, V_MNs)]
        
        # combine neural populations and prepare neural states and dynamical equations
        neurons = SNs + INs + MNs
        f = sy.Matrix([c.f for c in neurons]) 
        x = sy.Matrix([c.x for c in neurons])

        super(MechanicalFeedbackAndMutualInhibition, self).__init__(x=x, f=f)
