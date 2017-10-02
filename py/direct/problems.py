import PyKEP as pk
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

class point2orbit:

    '''
    Solves mass-optimal transfer from point to orbit with final anomoly free
    z = [T, mf, Mf, u]
    '''

    def __init__(
        self,
        r0,
        v0,
        nseg=50, mass=1000, tmax=0.1, isp=2500,
        elemf = [
            227943822376.03537,
            0.09339409892101332,
            0.032283207367640024,
            0.8649771996521327,
            5.000312830124232,
            0
        ]
        ):

        # intial and final keplerian elements
        self.r0 = r0
        self.v0 = v0
        self.elemf = elemf

        # number of segements
        self.nseg = nseg

        # nuclear electric spacecraft
        self.sc = pk.sims_flanagan.spacecraft(mass, tmax, isp)

        # low-thrust high-fidelity leg
        self.leg = pk.sims_flanagan.leg()
        self.leg.set_spacecraft(self.sc)
        self.leg.set_mu(pk.MU_SUN)
        self.leg.high_fidelity = True

    def fitness(self, z):

        # initial and final times
        t0 = pk.epoch(0)
        tf = pk.epoch(z[0])

        # final mass
        mf = z[1]

        # final mean anomoly
        self.elemf[5] = z[2]

        # control
        control = z[3:]

        # get final cartesian state
        rf, vf = pk.par2ic(self.elemf, pk.MU_SUN)

        # create spacecraft states
        x0 = pk.sims_flanagan.sc_state(self.r0, self.v0, self.sc.mass)
        xf = pk.sims_flanagan.sc_state(rf, vf, mf)

        # set leg
        self.leg.set(t0, x0, control, tf, xf)

        # compute equality constraints
        ceq = list(self.leg.mismatch_constraints())

        # nondimensionalise constraints
        ceq[0] /= pk.AU
        ceq[1] /= pk.AU
        ceq[2] /= pk.AU
        ceq[3] /= pk.EARTH_VELOCITY
        ceq[4] /= pk.EARTH_VELOCITY
        ceq[5] /= pk.EARTH_VELOCITY
        ceq[6] /= self.sc.mass

        # inequality constraints
        cineq = list(self.leg.throttles_constraints())

        return [-mf] + ceq + cineq

    def get_bounds(self):
        pi = 3.14159265359
        lb = [200, self.sc.mass/10, 0*pi, *(-1,-1,-1)*self.nseg]
        ub = [3000, self.sc.mass, 2*pi, *(1,1,1)*self.nseg]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def get_nic(self):
        return self.nseg

    def plot_traj(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # leg
        self.fitness(z)
        pk.orbit_plots.plot_sf_leg(self.leg, units=pk.AU, N=20, ax=axis)

        # initial and final times
        t0 = pk.epoch(0)
        tf = pk.epoch(z[0])

        # create fictitious keplerian planets
        kep0 = pk.planet.keplerian(t0, pk.ic2par(self.r0, self.v0, pk.MU_SUN))
        kepf = pk.planet.keplerian(tf, self.elemf)

        # plot planets
        pk.orbit_plots.plot_planet(kep0, t0, units=pk.AU, color=(0.8,0.8,1), ax=axis)
        pk.orbit_plots.plot_planet(kepf, tf, units=pk.AU, color=(0.8,0.8,1), ax=axis)

        # show the plot
        plt.show()

    def plot_control(self, z):
        u = z[3:]
        u = u.reshape((self.nseg, 3))
        u = norm(u, None, 1)
        plt.plot(u, "k.-")
        plt.show()

    def state_control(self, z):

        # make sure leg is set
        self.fitness(z)
        # get states and control
        t, r, v, m = self.leg.get_states()
        u = z[3:]
        # remove matchpoint duplicate
        t.pop(self.nseg)
        r.pop(self.nseg)
        v.pop(self.nseg)
        m.pop(self.nseg)
        # only keep midpoints
        t = t[1:self.nseg*2:2]
        r = r[1:self.nseg*2:2]
        v = v[1:self.nseg*2:2]
        m = m[1:self.nseg*2:2]
        # make numpy arrays
        t = np.asarray(t)
        r = np.asarray(r)
        v = np.asarray(v)
        m = np.asarray(m)
        u = np.asarray(u)
        # ensure correct dimension
        t = t.reshape((self.nseg, 1))
        r = r.reshape((self.nseg, 3))
        v = v.reshape((self.nseg, 3))
        m = m.reshape((self.nseg, 1))
        u = u.reshape((self.nseg, 3))
        # get control magnitude
        umag = norm(u, None, 1)
        umag = umag.reshape((self.nseg, 1))
        # concatentate the arrays
        xu = np.hstack((t, r, v, m, u, umag))
        return xu






class orbit2orbit:

    '''
    Solves mass-optimal transfer from orbit to orbit with mean anomolies free
    z = [T, mf, M0, Mf, u]
    '''

    def __init__(self,
        elem0 = [
            149598261129.93335,
            0.016711230601231957,
            2.640492490927786e-07,
            3.141592653589793,
            4.938194050401601,
            0
        ],
        elemf = [
            227943822376.03537,
            0.09339409892101332,
            0.032283207367640024,
            0.8649771996521327,
            5.000312830124232,
            0
        ],
        nseg=50, mass=1000, tmax=0.05, isp=2500):

        # initial and final keplerian elements
        self.elem0 = elem0
        self.elemf = elemf

        # number of segements
        self.nseg = nseg

        # nuclear electric spacecraft
        self.sc = pk.sims_flanagan.spacecraft(mass, tmax, isp)

        # low-thrust high-fidelity leg
        self.leg = pk.sims_flanagan.leg()
        self.leg.set_spacecraft(self.sc)
        self.leg.set_mu(pk.MU_SUN)
        self.leg.high_fidelity = True

    def fitness(self, z):

        # initial and final times
        t0 = pk.epoch(0)
        tf = pk.epoch(z[0])

        # final mass
        mf = z[1]

        # initial and final mean anomolies
        mu0 = z[2]
        muf = z[3]

        # controls
        controls = z[4:]

        self.elem0[5] = mu0
        self.elemf[5] = muf

        # get cartesian from keplerian
        r0, v0 = pk.par2ic(self.elem0, pk.MU_SUN)
        rf, vf = pk.par2ic(self.elemf, pk.MU_SUN)

        # create spacecraft states
        x0 = pk.sims_flanagan.sc_state(r0, v0, self.sc.mass)
        xf = pk.sims_flanagan.sc_state(rf, vf, mf)

        # set leg
        self.leg.set(t0, x0, controls, tf, xf)

        # compute equality constraints
        ceq = list(self.leg.mismatch_constraints())

        # nondimensionalise constraints
        ceq[0] /= pk.AU
        ceq[1] /= pk.AU
        ceq[2] /= pk.AU
        ceq[3] /= pk.EARTH_VELOCITY
        ceq[4] /= pk.EARTH_VELOCITY
        ceq[5] /= pk.EARTH_VELOCITY
        ceq[6] /= self.sc.mass

        # inequality constraints
        cineq = list(self.leg.throttles_constraints())

        return [-mf] + ceq + cineq

    def get_bounds(self):
        pi = 3.14159265359
        lb = [200, self.sc.mass/10, 0*pi, 0*pi, *(-1,-1,-1)*self.nseg]
        ub = [3000, self.sc.mass, 2*pi, 2*pi, *(1,1,1)*self.nseg]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def get_nic(self):
        return self.nseg

    def plot_traj(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # leg
        self.fitness(z)
        pk.orbit_plots.plot_sf_leg(self.leg, units=pk.AU, N=20, ax=axis)

        # initial and final times
        t0 = pk.epoch(0)
        tf = pk.epoch(z[0])

        # create fictitious keplerian planets
        kep0 = pk.planet.keplerian(t0, self.elem0)
        kepf = pk.planet.keplerian(tf, self.elemf)

        # plot planets
        pk.orbit_plots.plot_planet(kep0, t0, units=pk.AU, color=(0.8,0.8,1), ax=axis)
        pk.orbit_plots.plot_planet(kepf, tf, units=pk.AU, color=(0.8,0.8,1), ax=axis)

        # show the plot
        plt.show()

    def plot_control(self, z):
        u = z[4:]
        u = u.reshape((self.nseg, 3))
        u = norm(u, None, 1)
        plt.plot(u, "k.-")
        plt.show()

    def state_control(self, z):

        # make sure leg is set
        self.fitness(z)
        # get states and control
        t, r, v, m = self.leg.get_states()
        u = z[4:]
        # remove matchpoint duplicate
        t.pop(self.nseg)
        r.pop(self.nseg)
        v.pop(self.nseg)
        m.pop(self.nseg)
        # only keep midpoints
        t = t[1:self.nseg*2:2]
        r = r[1:self.nseg*2:2]
        v = v[1:self.nseg*2:2]
        m = m[1:self.nseg*2:2]
        # make numpy arrays
        t = np.asarray(t)
        r = np.asarray(r)
        v = np.asarray(v)
        m = np.asarray(m)
        u = np.asarray(u)
        # ensure correct dimension
        t = t.reshape((self.nseg, 1))
        r = r.reshape((self.nseg, 3))
        v = v.reshape((self.nseg, 3))
        m = m.reshape((self.nseg, 1))
        u = u.reshape((self.nseg, 3))
        # get control magnitude
        umag = norm(u, None, 1)
        umag = umag.reshape((self.nseg, 1))
        # concatentate the arrays
        xu = np.hstack((t, r, v, m, u, umag))
        return xu

class planet2planet(object):

    # z = [t0, T, mf, u]

    def __init__(
        self, p0="earth", pf="mars", mass=1000, tmax=0.3, isp=2500, nseg=50,
        t0lb=1000, t0ub=2000, Tlb=100, Tub=1000
    ):

        # planets
        self.p0 = pk.planet.jpl_lp(p0)
        self.pf = pk.planet.jpl_lp(pf)

        # spacecraft
        self.sc = pk.sims_flanagan.spacecraft(mass, tmax, isp)

        # direct leg
        self.leg = pk.sims_flanagan.leg()
        self.leg.set_spacecraft(self.sc)
        self.leg.set_mu(pk.MU_SUN)
        self.leg.high_fidelity = True

        # number of control segements
        self.nseg = nseg

        # bounds
        self.t0lb = t0lb
        self.t0ub = t0ub
        self.Tlb = Tlb
        self.Tub = Tub

    def fitness(self, z):

        # times
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # cartesian states
        r0, v0 = self.p0.eph(t0)
        rf, vf = self.pf.eph(tf)

        # final mass
        mf = z[2]

        # spacecraft states
        x0 = pk.sims_flanagan.sc_state(r0, v0, self.sc.mass)
        xf = pk.sims_flanagan.sc_state(rf, vf, mf)

        # set leg
        self.leg.set(t0, x0, z[3:], tf, xf)

        # equality constraints
        ceq = list(self.leg.mismatch_constraints())

        # nondimensionalise constraints
        ceq[0] /= pk.AU
        ceq[1] /= pk.AU
        ceq[2] /= pk.AU
        ceq[3] /= pk.EARTH_VELOCITY
        ceq[4] /= pk.EARTH_VELOCITY
        ceq[5] /= pk.EARTH_VELOCITY
        ceq[6] /= self.sc.mass

        # inequality constraints
        cineq = list(self.leg.throttles_constraints())

        return [-mf] + ceq + cineq

    def get_bounds(self):
        lb = [self.t0lb, self.Tlb, self.sc.mass/10, *(-1, -1, -1)*self.nseg]
        ub = [self.t0ub, self.Tub, self.sc.mass, *(1, 1, 1)*self.nseg]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def get_nic(self):
        return self.nseg

    def plot_traj(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # leg
        self.fitness(z)
        pk.orbit_plots.plot_sf_leg(self.leg, units=pk.AU, N=20, ax=axis)

        # initial and final times
        t0 = z[0]
        tf = z[0] + z[1]

        # plot planets
        pk.orbit_plots.plot_planet(self.p0, t0=pk.epoch(
            t0), units=pk.AU, ax=axis, color=(0.8, 0.8, 1))
        pk.orbit_plots.plot_planet(self.pf, t0=pk.epoch(
            tf), units=pk.AU, ax=axis, color=(0.8, 0.8, 1))

        # show the plot
        plt.show()

    def get_trajectory(self, z):

        # set leg
        self.fitness(z)

        # states
        t, r, v, m = self.leg.get_states()

        # controls
        u = z[3:]

        # remove matchpoint duplicate
        t.pop(self.nseg)
        r.pop(self.nseg)
        v.pop(self.nseg)
        m.pop(self.nseg)

        # only keep midpoints
        t = t[1:self.nseg*2:2]
        r = r[1:self.nseg*2:2]
        v = v[1:self.nseg*2:2]
        m = m[1:self.nseg*2:2]

        # make numpy arrays
        t = np.asarray(t)
        r = np.asarray(r)
        v = np.asarray(v)
        m = np.asarray(m)
        u = np.asarray(u)

        # ensure correct dimension
        t = t.reshape((self.nseg, 1))
        r = r.reshape((self.nseg, 3))
        v = v.reshape((self.nseg, 3))
        m = m.reshape((self.nseg, 1))
        u = u.reshape((self.nseg, 3))

        # control magnitude
        umag = norm(u, None, 1)
        umag = umag.reshape((self.nseg, 1))

        # concatentate arrays
        traj = np.hstack((t, r, v, m, umag, u))

        return traj
