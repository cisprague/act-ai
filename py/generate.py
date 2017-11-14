import PyKEP as pk
import pygmo as pg
import pygmo_plugins_nonfree as pg7
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import os
import cloudpickle as cp

# constants
pi = 3.14159265359

# algorithm
uda = pg7.snopt7(True, "/usr/lib/libsnopt7_c.so")
#uda.set_integer_option("Iterations limit", 40000)
uda.set_numeric_option("Major optimality tolerance", 1e-5)
uda.set_numeric_option("Major feasibility tolerance", 1e-10)

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
        Tlb    = 100,
        Tub    = 400,
        load   = True
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
            atol, rtol, (Tlb, Tub),
            freetime=True, alpha=0, bound=False, mu=mu
        )

        # pygmo problem
        prob = pg.problem(udp)

        # constraint tolerance
        prob.c_tol = [1e-6]*udp.get_nec()

        # solution filename
        fp = os.path.realpath(__file__)
        self.fp = os.path.split(fp)[0]
        fn = self.fp + "/../npy/indirect_or2or_" + o0 + "2" + of + ".npy"

        uda.set_integer_option("Major iterations limit", 100)
        algo = pg.algorithm(uda)

        # check for seed
        try:
            if load:
                # attempt to load decision vector
                znom = np.load(fn)
                print("Found nominal trajectory.")
                pop = pg.population(prob, 0)
                pop.push_back(znom)
                # polish solution incase of different atol and rtol
                pop = algo.evolve(pop)
            else:
                raise EnvironmentError

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

    def random_walk(self, state, npts=20, stepnom=0.1, alpha=0):

        # step record
        steps = []

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

        uda.set_integer_option("Major iterations limit", 30)
        algo = pg.algorithm(uda)

        for i in range(npts):

            # starting point specs
            z0   = np.copy(znom)
            x00  = np.copy(x0nom)
            # norms by which to perturb
            R = np.linalg.norm(x00[0:3])
            V = np.linalg.norm(x00[3:6])

            # optimise until feasible
            while True:

                # perturb state
                x01 = np.copy(x00)
                x01[0:3] += R*step*np.random.uniform(-1, 1, 3)
                x01[3:6] += V*step*np.random.uniform(-1, 1, 3)
                x01[6]   += x00[6]*step*np.random.uniform(-1, 1)

                # initialise point to orbit problem
                udp = pk.trajopt.indirect_pt2or(
                    x01, self.udpnom.elemf, self.udpnom.sc.mass,
                    self.udpnom.sc.thrust, self.udpnom.sc.isp,
                    self.udpnom.atol, self.udpnom.rtol,
                    self.udpnom.tof,
                    freetime=True, alpha=alpha, bound=True,
                    mu=self.udpnom.leg.mu
                )

                # pygmo problem
                prob = pg.problem(udp)

                # constraint tolerance
                prob.c_tol = [1e-8]*udp.get_nec()

                # population
                pop = pg.population(prob, 0)

                # use previous guess
                pop.push_back(z0)

                # optimise
                pop = algo.evolve(pop)

                # check feasibility
                if prob.feasibility_x(pop.champion_x):
                    print("Success on trajectory " + str(i) + " with step size " + str(step))
                    # set problem
                    udp.fitness(pop.champion_x)
                    # record decision chromosome
                    udp.z = pop.champion_x
                    # record trajectory
                    probs.append(udp)
                    # seed new decision
                    z0  = np.copy(pop.champion_x)
                    # seed new state
                    x00 = np.copy(x01)
                    # record the succesfull step size
                    steps.append(float(step))
                    # increase the perturbation size
                    #step = (np.average(steps, weights=np.linspace(0.1, 1, len(steps))) + step + stepnom)/3
                    step = (step + stepnom)/2

                    print("Perturbation size now " + str(step) + "\n")
                    # move to next state
                    break
                else:
                    print("Failure on trajectory " + str(i) + " with step size " + str(step))
                    # decrease perturbation size
                    if steps:
                        #step = (np.average(steps, weights=np.linspace(0.1, 1, len(steps))) + step + 0)/3
                        step /= 2
                    else:
                        step /= 2
                    print("Perturbation size now " + str(step) + "\n")
                    # try again
                    continue

        # save set probs
        fn = self.fp + "/../p/random_walk.p"
        cp.dump(probs, open(fn, "wb"))

        return probs

    def homotopy(self, prob):

        # extract nominal problem parametres
        alpha = 1
        x0    = prob.leg.x0
        z     = prob.z

        # algorithm parametres
        uda.set_integer_option("Major iterations limit", 20)
        algo = pg.algorithm(uda)

        # current highest feasible alpha
        alphastar = 0
        alphalim  = 0.99
        nfailed = 0

        # keep solving this shit until it works
        while True and nfailed<4:

            # initialise point to orbit problem
            udp = pk.trajopt.indirect_pt2or(
                x0, self.udpnom.elemf, self.udpnom.sc.mass,
                self.udpnom.sc.thrust, self.udpnom.sc.isp,
                self.udpnom.atol, self.udpnom.rtol,
                self.udpnom.tof,
                freetime=True, alpha=alpha, bound=True,
                mu=self.udpnom.leg.mu
            )

            # pygmo problem
            prob = pg.problem(udp)

            # constraint tolerance
            prob.c_tol = [1e-6]*udp.get_nec()

            # population
            pop = pg.population(prob, 0)
            pop.push_back(z)

            # solve
            print("Optimising with alpha = " + str(alpha) + "...")
            pop = algo.evolve(pop)

            # if it worked
            if prob.feasibility_x(pop.champion_x):

                print("Feasible, increasing alpha...\n")

                # set problem
                udp.fitness(pop.champion_x)
                # store decision chromosome
                udp.z = pop.champion_x
                # store the problem
                best_prob = udp

                # if finished
                if alpha == 1:
                    print("Achieved mass-optimal control! Sugoi desu ne!\n")
                    break

                # set new decision
                z = pop.champion_x

                # new best alpha
                alphastar = alpha

                # new alpha
                if alpha < alphalim:
                    alpha = (1 + alpha)/2
                elif alpha >= alphalim:
                    alpha = 1

                continue

            # if it failed
            else:
                print("Infeasible, decreasing alpha...")

                nfailed += 1
                print(str(nfailed) + " failed tries on this trajectory.")

                alpha = (alpha + alphastar)/2

                print()
                continue

        try:
            return best_prob

        except:
            pass

    def gen_quadratic_database(
        self,
        nnoms    = 10,
        nwalks   = [5]*10,
        nsteps   = [40]*10,
        stepnoms = [0.2]*10
    ):

        # nominal states
        noms = self.get_nominal_states(nnoms)

        # stage nominal state arg for each walk
        walkargs = list()

        # for each nominal state
        for ni in range(nnoms):

            # for each walk from that nominal state
            for wi in range(nwalks[ni]):
                walkargs.append((noms[ni], nsteps[ni], stepnoms[ni]))

        # start workers
        p = mp.Pool(4)

        # perform walks in parallel
        probs = p.map(lambda arg: self.random_walk(arg[0], arg[1], arg[2], alpha=0), walkargs)

        # 1d array of quadratically optimal problems with self.z
        probs = sum(probs, [])

        # save probs
        fn = self.fp + "/../p/random__qc_walks.p"
        cp.dump(probs, open(fn, "wb"))

        return probs

    def gen_mop_database(self, probs):

        # set up parallel workers
        p = mp.Pool(4)

        # convert all quadratic trajectories to mass-optimal ones in parallel
        probs = p.map(self.homotopy, probs)

        # save the probs
        fn = self.fp + "/../p/random_moc_walks.p"
        cp.dump(probs, open(fn, "wb"))

        return probs

    def gen_database(
        self,
        nnoms    = 10,
        nwalks   = [5]*10,
        nsteps   = [40]*10,
        stepnoms = [0.2]*10,
        search   = False
    ):

        if search:
            trajs = cp.load(open(self.fp + "/../p/random__qc_walks.p"))
        # generate quadratic trajectories in parallel
        trajs = self.gen_quadratic_database(nnoms, nwalks, nsteps, stepnoms)

        # optimise all those trajectories for mass
        trajs = self.gen_mop_database(trajs)

        return trajs

    def plot_trajs(self, probs):

        fig = plt.figure()
        axes = fig.gca(projection='3d')

        for prob in probs:
            prob.leg.plot_traj(axes, mark="k-", atol=self.udpnom.atol, rtol=self.udpnom.rtol)

        plt.show()

if __name__ == "__main__":

    # generator
    gen = generator(load = True)

    # generate database of mass-optimal trajectories in parallel
    nnodes = 10
    data = gen.gen_database(
        nnodes,
        [10]*nnodes,
        [40]*nnodes,
        [0.2]*nnodes
    )
