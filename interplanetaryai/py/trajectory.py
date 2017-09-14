import PyKEP as pk
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class orbit2orbit:

    def __init__(self, orbit0="earth", orbit1="mars", nseg=20, mass=1000, tmax=0.05, isp=2500):

        # get the planets
        p0 = pk.planet.jpl_lp(orbit0)
        p1 = pk.planet.jpl_lp(orbit1)
        self.p0 = p0
        self.p1 = p1

        # number control nodes
        self.nseg = nseg

        # start and duration
        t0 = 0
        T = 10000

        # epochs
        t1 = pk.epoch(t0 + T)
        t0 = pk.epoch(t0)

        # keplerian elements of planets
        self.orbit0 = list(p0.osculating_elements(t0))
        self.orbit1 = list(p1.osculating_elements(t1))

        # instantiate spacecraft
        self.sc = pk.sims_flanagan.spacecraft(mass, tmax, isp)

        # final mass
        mf = mass

        # initial and final spacecraft states
        r0, v0 = p0.eph(t0)
        r1, v1 = p1.eph(t1)
        x0 = pk.sims_flanagan.sc_state(r0, v0, self.sc.mass)
        x1 = pk.sims_flanagan.sc_state(r1, v1, self.sc.mass)

        # gravitational parametre
        mu = pk.MU_SUN

        # controls seed
        controls = [1, 0, 0]*self.nseg

        # instantiate leg
        self.leg = pk.sims_flanagan.leg(t0, x0, controls, t1, x1, self.sc, mu)
        self.leg.high_fidelity = True

    def fitness(self, z):

        # unpack decision vector
        t0 = pk.epoch(z[0])
        t1 = pk.epoch(z[0] + z[1])
        mf = z[2]
        mu0 = z[3]
        mu1 = z[4]

        # controls decisions
        condec = np.asarray(z[5:])
        condec = np.reshape(condec, (self.nseg, 3))
        controls = np.zeros_like(condec)
        a = np.sqrt(1-condec[:,2]**2)
        controls[:,0] = np.cos(condec[:,1])*a
        controls[:,1] = np.sin(condec[:,1])*a
        controls[:,2] = condec[:,2]
        condec = np.reshape(condec[:,0],(self.nseg, 1))
        control = condec*controls

        # now we have the real controls
        controls = list(np.reshape(controls, self.nseg*3))

        # change the true anomolies of orbits
        self.orbit0[5] = mu0
        self.orbit1[5] = mu1

        # create keplerian planet
        p0 = pk.planet.keplerian(t0, self.orbit0)
        p1 = pk.planet.keplerian(t1, self.orbit1)

        # get cartesian
        r0, v0 = p0.eph(t0)
        r1, v1 = p1.eph(t1)

        # states
        x0 = pk.sims_flanagan.sc_state(r0, v0, self.sc.mass)
        x1 = pk.sims_flanagan.sc_state(r1, v1, mf)

        # set leg
        self.leg.set(t0, x0, controls, t1, x1)

        # compute eq constraints
        ceq = list(self.leg.mismatch_constraints())

        # nondimensionalise constraints
        ceq[0] /= pk.AU
        ceq[1] /= pk.AU
        ceq[2] /= pk.AU
        ceq[3] /= pk.EARTH_VELOCITY
        ceq[4] /= pk.EARTH_VELOCITY
        ceq[5] /= pk.EARTH_VELOCITY
        ceq[6] /= self.sc.mass

        return [-mf] + ceq

    def get_bounds(self):
        pi = 3.14159265359
        lb = [6000, 10, self.sc.mass/20, 0, 0, *(0,0,-1)*self.nseg]
        ub = [14600, 3000, self.sc.mass, 2*pi, 2*pi, *(1,2*pi,1)*self.nseg]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def plot(self, z):

        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # leg
        pk.orbit_plots.plot_sf_leg(self.leg, units=pk.AU, N=20, ax=axis)

        # epochs
        t0 = pk.epoch(z[0])
        t1 = pk.epoch(z[1])

        # get planet orbital elements
        orbit0 = list(self.p0.osculating_elements(t0))
        orbit1 = list(self.p1.osculating_elements(t1))

        # replace true anomoly
        orbit0[5] = z[3]
        orbit1[5] = z[4]

        # create Keplerian planet
        p0 = pk.planet.keplerian(t0, orbit0)
        p1 = pk.planet.keplerian(t1, orbit1)

        # plot planets
        pk.orbit_plots.plot_planet(p0, t0, units=pk.AU, color=(0.8,0.8,1), ax=axis)
        pk.orbit_plots.plot_planet(p1, t1, units=pk.AU, color=(0.8,0.8,1), ax=axis)

        # show the plot
        plt.show()


def test_plot():
    prob = orbit2orbit(nseg=20)
    z = [0, 1000, 900, 0, 1, *(0,0,0)*prob.nseg]
    prob.plot(z)

def test_optimisation():
    import pygmo as pg
    import pygmo_plugins_nonfree as pg7

    prob = pg.problem(orbit2orbit(nseg=20))

    uda = pg7.snopt7(True, "/usr/lib/libsnopt7_c.so")
    algo = pg.algorithm(uda)
    algo.set_verbosity(10)
    uda = pg.mbh(algo, stop=10)
    algo = pg.algorithm(uda)
    algo.set_verbosity(10)

    pop = pg.population(prob, 1)
    pop = algo.evolve(pop)

    prob.plot(pop.champion_x)


if __name__ == "__main__":

    prob = orbit2orbit()

    z = [0, 1000, 920, 0, 0.2, *(0.5, 0.8, 0.1)*prob.nseg]

    prob.fitness(z)
