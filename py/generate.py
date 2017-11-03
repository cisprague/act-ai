import PyKEP as pk
import pygmo as pg
import pygmo_plugins_nonfree as pg7
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

# constants
pi = 3.14159265359

# algorithm
uda = pg7.snopt7(True, "/usr/lib/libsnopt7_c.so")
uda.set_integer_option("Major iterations limit", 100)
#uda.set_integer_option("Iterations limit", 40000)
uda.set_numeric_option("Major optimality tolerance", 1e-2)
uda.set_numeric_option("Major feasibility tolerance", 1e-8)
algo = pg.algorithm(uda)

class generator(object):

    def __init__(
        self,
        o0     = "earth",
        of     = "mars",
        mu     = pk.MU_SUN,
        mass   = 1000,
        thrust = 0.3,
        isp    = 2500,
        atol   = 1e-12,
        rtol   = 1e-12,
        Mlb    = -4*pi,
        Mub    = 4*pi,
        Tlb    = 100,
        Tub    = 400
    ):

        # planets
        p0 = pk.planet.jpl_lp(o0)
        pf = pk.planet.jpl_lp(of)

        # Keplerian elements
        el0 = np.array(p0.osculating_elements(pk.epoch(0)))
        elf = np.array(pf.osculating_elements(pk.epoch(0)))

        # indirect quadratic orbit to orbit
        udp = pk.trajopt.indirect_or2or(
            el0, elf, mass, thrust, isp,
            atol, rtol, Tlb, Tub, Mlb, Mub, Mlb, Mub,
            freetime=True, alpha=0, bound=False, mu=mu
        )

        # pygmo problem
        prob = pg.problem(udp)

        # constraint tolerance
        prob.c_tol = [1e-6]*udp.get_nec()

        # solution filename
        fn = os.path.realpath(__file__)
        fn = os.path.split(fn)[0]
        fn += "/indirect_or2or_" + o0 + "2" + of + ".npy"

        # check for seed
        try:

            # attempt to load decision vector
            znom = np.load(fn)
            print("Found nominal trajectory.")
            pop = pg.population(prob, 0)
            pop.push_back(znom)
            # polish solution incase of different atol and rtol
            pop = algo.evolve(pop)

        # if seed not found
        except:

            # optimise until feasible
            while True:
                pop = pg.population(prob, 0)
                zguess = np.hstack(([np.random.uniform(Tlb, Tub)], np.random.randn(9)))
                pop.push_back(zguess)
                pop = algo.evolve(pop)
                if prob.feasibility_x(pop.champion_x):
                    print("Generated feasible solution!")
                    znom = pop.champion_x
                    np.save(fn, znom)
                    break

        # store stuff for later
        udp.fitness(znom)
        self.udpnom  = udp
        self.znom    = znom
        self.trajnom = udp.leg.get_states(atol, rtol)

    def get_nominal_states(self, n):

        # number of nominal points
        npts = self.trajnom.shape[0]

        # nominal trajectory indices
        i = np.linspace(0, npts, n, endpoint=False, dtype=int)

        # return sampled states
        return self.trajnom[i]

    def random_walk_quadratic(self, state, npts=20, stepnom=0.1):

        # trajectory record
        probs = list()

        # nominal time of flight
        Tnom = self.znom[0] - state[0]

        # nominal arrival eccentric anomoly
        Efnom = self.znom[2]

        # nominal departure costates
        l0nom = state[8:15]

        # nominal point to orbit decision
        znom = np.hstack(([Tnom, Efnom], l0nom))

        # nominal state
        x0nom = state[1:8]
        step = np.copy(stepnom)

        for i in range(npts):

            # starting point specs
            z0   = np.copy(znom)
            x00  = np.copy(x0nom)

            # optimise until feasible
            while True:

                # perturb state
                x01 = x00*(1 + step*np.random.uniform(-1, 1, 7))

                # initialise point to orbit problem
                udp = pk.trajopt.indirect_pt2or(
                    x01, self.udpnom.elemf, self.udpnom.sc.mass,
                    self.udpnom.sc.thrust, self.udpnom.sc.isp,
                    self.udpnom.atol, self.udpnom.rtol,
                    self.udpnom.Tlb, self.udpnom.Tub,
                    self.udpnom.Mflb, self.udpnom.Mfub,
                    freetime=True, alpha=0, bound=False,
                    mu=self.udpnom.leg.mu
                )

                # pygmo problem
                prob = pg.problem(udp)

                # constraint tolerance
                prob.c_tol = [1e-6]*udp.get_nec()

                # population
                pop = pg.population(prob, 0)

                # use previous guess
                pop.push_back(z0)

                # optimise
                pop = algo.evolve(pop)

                # check feasibility
                if prob.feasibility_x(pop.champion_x):
                    # set problem
                    udp.fitness(pop.champion_x)
                    # record trajectory
                    probs.append(udp)
                    # seed new decision
                    z0  = np.copy(pop.champion_x)
                    # seed new state
                    x00 = np.copy(x01)
                    # increase the perturbation size
                    step += (stepnom - step)/2

                    print("Success on trajectory " + str(i))
                    print("Perturbation size now " + str(step) + "\n")
                    # move to next state
                    break
                else:
                    # decrease perturbation size
                    step /= 2

                    print("Failure on trajectory " + str(i))
                    print("Perturbation size now " + str(step) + "\n")
                    # try again
                    continue

        return probs

    def plot_trajs(self, probs):

        fig = plt.figure()
        axes = fig.gca(projection='3d')

        for prob in probs:
            prob.leg.plot_traj(axes, mark="k-", atol=self.udpnom.atol, rtol=self.udpnom.rtol)

        plt.show()












if __name__ == "__main__":

    # create generator object
    gen = generator(atol=1e-12, rtol=1e-12)

    # nominal state
    s0 = gen.get_nominal_states(10)[0]

    # random walk
    gen.random_walk_quadratic(s0)
