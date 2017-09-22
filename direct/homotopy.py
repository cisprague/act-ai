import pygmo as pg
import pygmo_plugins_nonfree as pg7
import PyKEP as pk
import numpy as np
from trajectory import *
from data import *

class generator(object):

    def __init__(self, nseg=50, mass=1000, tmax=0.1, isp=2500, seed=None):

        # data record
        self.legs = list()
        self.data = list()

        # configuration
        self.nseg = nseg
        self.mass = mass
        self.tmax = tmax
        self.isp = isp

        # setup algorithm
        uda = pg7.snopt7(True, "/usr/lib/libsnopt7_c.so")
        uda.set_integer_option("Major iterations limit", 4000)
        uda.set_integer_option("Iterations limit", 40000)
        uda.set_numeric_option("Major optimality tolerance", 1e-3)
        uda.set_numeric_option("Major feasibility tolerance", 1e-10)
        self.algo = pg.algorithm(uda)

        # solve orbit to orbit problem
        udp = orbit2orbit(nseg=50, mass=1000, tmax=0.1, isp=2500)
        prob = pg.problem(udp)
        prob.c_tol = [1e-8]*(7 + self.nseg)

        try:
            seed = np.load(seed)
            pop = pg.population(prob, 0)
            pop.push_back(seed)
        except:
            pop = pg.population(prob, 1)
            while not prob.feasibility_x(pop.champion_x):
                pop = pg.population(prob, 1)
                pop = self.algo.evolve(pop)

        # make a new decision vector
        T = pop.champion_x[0]
        mf = pop.champion_x[1]
        M0 = pop.champion_x[2]
        Mf = pop.champion_x[3]
        u = pop.champion_x[4:]
        self.guess = np.hstack(([T, mf, Mf], u))

        # store nominal cartesian state
        r0, v0 = pk.par2ic(udp.elem0, pk.MU_SUN)
        self.r0 = np.asarray(r0)
        self.v0 = np.asarray(v0)

        # store data of the nominal trajectory
        self.legs.append(udp.leg)
        self.data.append(udp.state_control(pop.champion_x))

    def homotopy(self):
        r = self.r0
        v = self.v0









if __name__ == "__main__":
    gen = generator()
