from spacecraft import spacecraft
from indirect.leg import leg

import PyKEP as pk
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class orbit2orbit(object):

    def __init__(
        self,
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
        mass=1000, tmax=0.05, isp=2500
    ):

        # initial and final Keplerian elements
        self.elem0 = elem0
        self.elemf = elemf

        # spacecraft
        self.sc = spacecraft(mass=mass, tmax=tmax, isp=isp)

        # indirect leg
        self.leg = leg(spacecraft=self.sc)

    def fitness(self, z):

        # initial and final times
        t0 = 0
        tf = z[0]

        # initial and final mean anomolies
        M0 = z[1]
        Mf = z[2]

        # initial costate variables
        l0 = np.asarray(z[3:])

        # set Keplerian elements
        self.elem0[5] = M0
        self.elemf[5] = Mf

        # compute Cartesian states
        r0, v0 = pk.par2ic(self.elem0, pk.MU_SUN)
        rf, vf = pk.par2ic(self.elemf, pk.MU_SUN)
        r0 = np.asarray(r0)
        v0 = np.asarray(v0)
        rf = np.asarray(rf)
        vf = np.asarray(vf)

        # set the indirect leg
        self.leg.set(t0, r0, v0, l0, tf, rf, vf)

        # equality constraints
        ceq = self.leg.mismatch_constraints(atol=1e-4, rtol=1e-5)

        # get final mass
        mf = self.leg.trajectory[-1, 6]

        return np.hstack(([-mf], ceq))

    def get_bounds(self):
        pi = 3.14159265359
        lb = [1.728e7, 0.0, 0.0, *[-1e5]*7]
        ub = [2.592e8, 2*pi, 2*pi, *[1e5]*7]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def plot_traj(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # make sure parametres are set
        self.fitness(z)

        # initial and final times in days
        t0 = pk.epoch(0)
        tf = pk.epoch(z[0]/60/60/24)

        # create Keplerian planets
        kep0 = pk.planet.keplerian(t0, self.elem0)
        kepf = pk.planet.keplerian(tf, self.elemf)

        # plot planets
        pk.orbit_plots.plot_planet(kep0, t0, units=pk.AU, color=(0.8,0.8,1), ax=axis)
        pk.orbit_plots.plot_planet(kepf, tf, units=pk.AU, color=(0.8,0.8,1), ax=axis)

        # plot trajectory
        traj = self.leg.trajectory
        axis.plot(traj[:,0], traj[:,1], traj[:,2], "k.-")

        # show plot
        plt.show()


class planet2planet(object):

    # z = [t0, T, costates], [sec, sec, nondim]

    def __init__(self, p0="earth", pf="mars", mass=1000, tmax=0.05, isp=1000, atol=1e-5, rtol=1e-5):

        # planets
        self.p0 = pk.planet.jpl_lp(p0)
        self.pf = pk.planet.jpl_lp(pf)

        # spacecraft
        self.sc = spacecraft(mass=mass, tmax=tmax, isp=isp)

        # indirect leg
        self.leg = leg(self.sc)

        # integration parametres
        self.atol = atol
        self.rtol = rtol

    def fitness(self, z):

        # times
        t0 = z[0]
        tf = t0 + z[1]

        # initial costates
        l0 = np.asarray(z[2:])

        # cartesian states
        r0, v0 = np.asarray(self.p0.eph(t0))
        rf, vf = np.asarray(self.pf.eph(tf))

        # set leg
        self.leg.set(t0*24*60*60, r0, v0, l0, tf*24*60*60, rf, vf)

        # propagate leg
        ceq = self.leg.mismatch_constraints(atol=self.atol, rtol=self.rtol)

        # get final state
        mf = self.leg.trajectory[-1, 6]

        return np.hstack(([-mf], ceq))

    def get_bounds(self):
        lb = [1000, 200, *[-1e2]*7]
        ub = [4000, 5000, *[1e2]*7]
        return (lb, ub)

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 7

    def plot_traj(self, z):

        # set up figure
        fig = plt.figure()
        axis = fig.gca(projection='3d')

        # sun
        axis.scatter([0], [0], [0], color='y')

        # set parametres
        self.fitness(z)

        # times
        t0 = z[0]
        tf = z[0] + z[1]

        # plot planets
        pk.orbit_plots.plot_planet(self.p0, t0=pk.epoch(t0), units=pk.AU, ax=axis, color=(0.8, 0.8, 1))
        pk.orbit_plots.plot_planet(self.pf, t0=pk.epoch(tf), units=pk.AU, ax=axis, color=(0.8, 0.8, 1))

        # plot trajectory
        traj = self.leg.trajectory
        axis.plot(traj[:,0], traj[:,1], traj[:,2], "k.-")

        # show plot
        plt.show()
