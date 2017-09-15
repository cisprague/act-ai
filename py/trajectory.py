import PyKEP as pk
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class orbit2orbit:

    def __init__(self, orbit0="earth", orbitf="mars", nseg=50, mass=1000, tmax=0.05, isp=2500):

        # initial and final planets
        self.p0 = pk.planet.jpl_lp(orbit0)
        self.pf = pk.planet.jpl_lp(orbitf)

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
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # final mass
        mf = z[2]

        # initial and final mean anomolies
        mu0 = z[3]
        muf = z[4]

        # controls
        controls = z[5:]

        # keplerian elements of planets
        elem0 = list(self.p0.osculating_elements(t0))
        elemf = list(self.pf.osculating_elements(tf))

        # replace mean anomolies
        elem0[5] = mu0
        elemf[5] = muf

        # create fictitious keplerian planets
        kep0 = pk.planet.keplerian(t0, elem0)
        kepf = pk.planet.keplerian(tf, elemf)

        # get cartesian states
        r0, v0 = kep0.eph(t0)
        rf, vf = kepf.eph(tf)

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
        lb = [6570, 1825, self.sc.mass/10, 0, 0, *(0,0,0)*self.nseg]
        ub = [10220, 7300, self.sc.mass, 2*pi, 2*pi, *(1,1,1)*self.nseg]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def get_nic(self):
        return self.nseg

    def plot(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # leg
        self.fitness(z)
        pk.orbit_plots.plot_sf_leg(self.leg, units=pk.AU, N=20, ax=axis)

        # initial and final times
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # planet keplerian elements
        elem0 = list(self.p0.osculating_elements(t0))
        elemf = list(self.pf.osculating_elements(tf))

        # replace mean anomolies
        elem0[5] = z[3]
        elemf[5] = z[4]

        # create fictitious keplerian planets
        kep0 = pk.planet.keplerian(t0, elem0)
        kepf = pk.planet.keplerian(tf, elemf)

        # plot planets
        pk.orbit_plots.plot_planet(kep0, t0, units=pk.AU, color=(0.8,0.8,1), ax=axis)
        pk.orbit_plots.plot_planet(kepf, tf, units=pk.AU, color=(0.8,0.8,1), ax=axis)

        # show the plot
        plt.show()

class p2p:

    def __init__(self, p0="earth", pf="mars", nseg=50, mass=1000, tmax=0.10, isp=2500):

        # initial and final planets
        self.p0 = pk.planet.jpl_lp(p0)
        self.pf = pk.planet.jpl_lp(pf)

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
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # final mass
        mf = z[2]

        # controls
        controls = z[3:]

        # initial and final states
        r0, v0 = self.p0.eph(t0)
        rf, vf = self.pf.eph(tf)

        # spacecraft state
        x0 = pk.sims_flanagan.sc_state(r0, v0, self.sc.mass)
        xf = pk.sims_flanagan.sc_state(rf, vf, mf)

        # set leg
        self.leg.set(t0, x0, controls, tf, xf)

        # euality constraints
        ceq = list(self.leg.mismatch_constraints())

        # nondimensionalise
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
        lb = [0, 200, self.sc.mass/10, *(-1,-1,-1)*self.nseg]
        ub = [1000, 1000, self.sc.mass, *(1,1,1)*self.nseg]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def get_nic(self):
        return self.nseg

    def plot(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # leg
        self.fitness(z)
        pk.orbit_plots.plot_sf_leg(self.leg, units=pk.AU, N=20, ax=axis)

        # initial and final times
        t0 = pk.epoch(z[0])
        tf = pk.epoch(z[0] + z[1])

        # plot planets
        pk.orbit_plots.plot_planet(self.p0, t0, units=pk.AU, color=(0.8,0.8,1), ax=axis)
        pk.orbit_plots.plot_planet(self.pf, tf, units=pk.AU, color=(0.8,0.8,1), ax=axis)

        # show the plot
        plt.show()
