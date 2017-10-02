import numpy as np
from scipy.integrate import ode, odeint
from constants import MU_SUN
from indirect.dynamics import dynamics


class leg(object):

    def __init__(self, spacecraft, mu=MU_SUN):

        # spacecraft
        self.spacecraft = spacecraft

        # central body standard gravitational parametre
        self.mu = mu

        # dynamics
        self.dynamics = dynamics(spacecraft, mu)

        # integrator
        self.integrator = ode(
            lambda t, fs: self.dynamics.eom_fullstate(fs),
            lambda t, fs: self.dynamics.eom_fullstate_jac(fs)
        )


    def recorder(self, t, fs):

        # append time
        self.t = np.append(self.t, t)

        # append fullstate
        self.trajectory = np.vstack((self.trajectory, fs))


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

    def propagate(self, atol=1e-5, rtol=1e-5, adapt=True, npts=1000):

        # nondimensionalise state
        r0 = self.r0/self.dynamics.L
        v0 = self.v0/self.dynamics.V
        m0 = self.spacecraft.mass/self.dynamics.M

        # create nondimensional fullstate
        fs0 = np.hstack((r0, v0, [m0], self.l0))

        # nondimensionalise times
        t0 = self.t0/self.dynamics.T
        tf = self.tf/self.dynamics.T

        if adapt:

            # clear trajectory history
            self.t = np.empty((1,0), dtype=np.float64)
            self.trajectory = np.empty((0, 14), dtype=np.float64)

            # set integration method
            self.integrator.set_integrator("dopri5", atol=atol, rtol=rtol)

            # set recorder
            self.integrator.set_solout(self.recorder)


            # set initial conditions
            self.integrator.set_initial_value(fs0, t0)

            # numerically integrate
            self.integrator.integrate(tf)

        elif not adapt:

            # time sequence
            self.t = np.linspace(t0, tf, npts, dtype=np.float64)

            self.trajectory = odeint(
                lambda fs, t: self.dynamics.eom_fullstate(fs),
                fs0,
                self.t,
                Dfun = lambda fs, t: self.dynamics.eom_fullstate_jac(fs),
                atol = atol,
                rtol = rtol
            )

    def mismatch_constraints(self, atol=1e-5, rtol=1e-5, adapt=True, npts=1000):

        # propagate trajectory
        self.propagate(atol=atol, rtol=rtol, adapt=adapt, npts=npts)

        # final conditions
        rf = self.trajectory[-1, 0:3]
        vf = self.trajectory[-1, 3:6]
        lmf = self.trajectory[-1, 13]

        # compute nondimensional arrival mismatch
        drf = rf - self.rf/self.dynamics.L
        dvf = vf - self.vf/self.dynamics.V

        # compute Hamiltonian
        H = self.dynamics.hamiltonian(self.trajectory[-1])

        # create equality constraints
        ceq = np.hstack((drf, dvf, [lmf], [H]))

        return ceq

    def get_trajectory(self, atol=1e-12, rtol=1e-12, adapt=False, npts=1000):

        # traj = [t, x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm, u, ux, uy, uz]
        # traj.shape = (npts, 19)

        # propagate trajectory
        self.propagate(atol=atol, rtol=rtol, adapt=adapt, npts=npts)

        # get times
        t = self.t.reshape(self.t.size, 1)
        # redimensionalise times
        t *= self.dynamics.T

        # get controls
        u = np.asarray([self.dynamics.pontryagin(fs) for fs in self.trajectory])

        # get Hamiltonian
        H = np.asarray([self.dynamics.hamiltonian(fs) for fs in self.trajectory])
        H = H.reshape(H.size, 1)

        # get trajectory
        traj = self.trajectory
        # redimensionalise trajectory
        traj[:, 0:3] *= self.dynamics.L
        traj[:, 3:6] *= self.dynamics.V
        traj[:, 6] *= self.dynamics.M

        # assemble full trajectory history
        traj = np.hstack((t, traj, u, H))

        return traj
