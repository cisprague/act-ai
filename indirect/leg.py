import numpy as np
from scipy.integrate import odeint
from constants import MU_SUN
from dynamics import dynamics


class leg(object):

    def __init__(self, spacecraft, mu=MU_SUN):

        # spacecraft
        self.spacecraft = spacecraft

        # central body standard gravitational parametre
        self.mu = mu

        # dynamics
        self.dynamics = dynamics(spacecraft, mu)

        # departure conditions
        self.t0 = None
        self.r0 = None
        self.v0 = None
        self.l0 = None
        self.fs0 = None

        # arrival conditions
        self.tf = None
        self.rf = None
        self.vf = None

        # fullstates
        self.trajectory = None

    def set(self, t0, r0, v0, l0, tf, rf, vf):

        # departure
        self.t0 = t0
        self.r0 = r0
        self.v0 = v0
        self.l0 = l0

        # arrival
        self.tf = tf
        self.rf = rf
        self.vf = vf

    def recorder(self, t, x):
        row = np.hstack(([t], x))
        self.trajectory = np.vstack((self.trajectory, row))

    def propagate(self, atol=1e-5, rtol=1e-5, npts=1000):

        # assemble intial fullstate
        fs0 = np.hstack((self.r0/self.dynamics.L, self.v0/self.dynamics.V, [self.spacecraft.mass/self.dynamics.M], self.l0))

        # times
        t = np.linspace(self.t0/self.dynamics.T, self.tf/self.dynamics.T, num=npts)

        # numerically integrate
        self.trajectory = odeint(
            lambda x, t: self.dynamics.eom_fullstate(x),
            fs0,
            t,
            #Dfun = lambda x, t: self.dynamics.eom_fullstate_jac(x),
            atol = atol,
            rtol = rtol,
            hmax = 0.01,
            hmin = 1e-12
        )

    def mismatch_constraints(self, atol=1e-5, rtol=1e-5, npts=1000):

        # propagate trajectory
        self.propagate(atol=atol, rtol=rtol, npts=npts)

        # final conditions
        rf = self.trajectory[-1, 0:3]
        vf = self.trajectory[-1, 3:6]
        lmf = self.trajectory[-1, 13]

        # compute mismatch
        ceq = np.hstack((rf - self.rf/self.dynamics.L, vf - self.vf/self.dynamics.V, [lmf]))

        return ceq
