import PyKEP as pk
import pygmo as pg
import pygmo_plugins_nonfree as pg7
import numpy as np
import matplotlib.pyplot as plt


# algorithm
uda = pg7.snopt7(True, "/usr/lib/libsnopt7_c.so")
uda.set_numeric_option("Major optimality tolerance", 1e-6)
uda.set_numeric_option("Major feasibility tolerance", 1e-10)
uda.set_integer_option("Major iterations limit", 300)
algo = pg.algorithm(uda)
pi = 3.1459

"""uda = pg.ipopt()
uda.set_integer_option("print_level", 5)
uda.set_numeric_option("tol", 1e-5)
uda.set_numeric_option("dual_inf_tol", 1e-4)
uda.set_numeric_option("constr_viol_tol", 1e-5)
uda.set_numeric_option("compl_inf_tol", 1e-4)
algo = pg.algorithm(uda)"""




def indirect_or2or(alpha, bound):

    # planets
    p0 = pk.planet.jpl_lp("earth")
    pf = pk.planet.jpl_lp("mars")

    # Keplerian elements
    el0 = np.array(p0.osculating_elements(pk.epoch(0)))
    elf = np.array(pf.osculating_elements(pk.epoch(0)))

    # spacecraft
    mass   = 1000
    thrust = 0.3
    isp    = 2500

    # tolerances
    atol = 1e-10
    rtol = 1e-10

    # flight duration bounds
    Tlb = 200
    Tub = 1000

    # mean anomoly bounds
    Mlb = -4*pi
    Mub = 4*pi

    # durations
    times = np.linspace(Tlb, Tub, 100)

    # trajectories
    trajs = list()

    # for every time
    for T in times:

        print("Creating data points for T = " + str(T))

        # create orbit to orbit problem
        udp = pk.trajopt.indirect_or2or(
            el0, elf, mass, thrust, isp, atol, rtol,
            T, T, Mlb, Mub, Mlb, Mub, alpha=alpha, bound=bound,
            freetime=False
        )

        # pygmo problem
        prob = pg.problem(udp)

        # constraint tolerance
        prob.c_tol = [1e-6]*udp.get_nec()

        # maximum optimisation attempts
        niter = 500

        # number of samples
        ntraj = 1
        nfeas = 0

        # for every iteration (if needed)
        for i in range(niter):

            print("Iteration " + str(i))

            # use previous solution if possible
            try:

                print("Previous solution found!")

                # mean anomolies
                M0 = z[1]
                Mf = z[2]

                # random costates
                l0 = z[3:]*(1 + 0.01*np.random.uniform(-1, 1, 7))

                # guessed decision chromosome
                #z = z*(1 + 0.01*np.random.uniform(-1, 1, len(z)))
                z = np.hstack(([T, M0, Mf], l0))

                # population
                pop = pg.population(prob, 0)
                pop.push_back(z)

            # completely random guess if no previous solution
            except:

                print("No previous solution; creating randomly.")

                # random dpopulation
                pop = pg.population(prob, 0)
                pop.push_back(np.hstack(([T, -1, 1], np.random.randn(7))))

            # optimise trajectory
            pop = algo.evolve(pop)

            # if the solution is feasible
            if prob.feasibility_x(pop.champion_x):

                print("Solution is feasible!")

                # save the decision chromosome
                z = pop.champion_x

                # set problem
                udp.fitness(z)

                # trajectory data
                data = udp.leg.get_states(atol, rtol)

                # save decision and data
                trajs.append((z, data))

                # add to counter
                nfeas += 1

                print("Now have " + str(nfeas) + " feasible.")

            # if solution infeasible
            else:

                print("Solution not feasible...")

                # continue to next iteration
                continue

            # if enough feasible trajectories for this time
            if nfeas == ntraj:

                print("Found enough solutions!")

                # quit iteration loop
                break

            # if need more feasible trajectories for this time
            else:
                print("Need more solutions")
                continue

    # create plot figure
    plt.figure()

    # for every trajectory in database
    for traj in trajs:

        # trajectory
        traj = traj[1]

        # flight duration
        T = traj[-1, 0] - traj[0, 0]

        # final mass
        mf = traj[-1, 7]

        # plot data point
        plt.scatter(T, mf)

    # show the plot
    plt.show()


def direct_or2or(nseg):

    # planets
    p0 = pk.planet.jpl_lp("earth")
    pf = pk.planet.jpl_lp("mars")

    # Keplerian elements
    el0 = np.array(p0.osculating_elements(pk.epoch(0)))
    elf = np.array(pf.osculating_elements(pk.epoch(0)))

    # spacecraft
    mass = 1000
    thrust = 0.3
    isp = 2500

    # flight duration bounds
    Tlb = 200
    Tub = 1000

    # mean anomoly bounds
    Mlb = -4*pi
    Mub = 4*pi

    # durations
    times = np.linspace(Tlb, Tub, 10)

    # trajectories
    trajs = list()

    # for every time
    for T in times:

        print("Creating data points for T = " + str(T))

        # create orbit to orbit problem
        udp = pk.trajopt.direct_or2or(
            el0, elf, mass, thrust, isp, nseg, T, T+.000001, Mlb, Mub, Mlb, Mub, quad=True
        )

        # pygmo problem
        prob = pg.problem(udp)

        # constraint tolerance
        prob.c_tol = [1e-6]*(udp.get_nec() + udp.get_nic())

        # maximum optimisation attempts
        niter = 100

        # number of samples
        ntraj = 3
        nfeas = 0

        # for every iteration (if needed)
        for i in range(niter):

            print("Iteration " + str(i))

            # population
            pop = pg.population(prob, 1)

            # optimise trajectory
            pop = algo.evolve(pop)

            # if the solution is feasible
            if prob.feasibility_x(pop.champion_x):

                print("Solution is feasible!")

                # save the decision chromosome
                z = pop.champion_x

                # trajectory data
                data = udp.get_traj(z)

                # save decision and data
                trajs.append((z, data))

                # add to counter
                nfeas += 1

                print("Now have " + str(nfeas) + " feasible.")

            # if solution infeasible
            else:

                print("Solution not feasible...")

                # continue to next iteration
                continue

            # if enough feasible trajectories for this time
            if nfeas == ntraj:

                print("Found enough solutions!")

                # quit iteration loop
                break

            # if need more feasible trajectories for this time
            else:
                print("Need more solutions")
                continue

    # create plot figure
    plt.figure()

    # for every trajectory in database
    for traj in trajs:

        # trajectory
        traj = traj[1]

        # flight duration
        T = traj[-1, 0] - traj[0, 0]

        # final mass
        mf = traj[-1, 7]

        # plot data point
        plt.scatter(T, mf)

    # show the plot
    plt.show()








if __name__ == "__main__":
    indirect_or2or(0, False)
