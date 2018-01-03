# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

from pathos.multiprocessing import Pool
import numpy as np, matplotlib.pyplot as plt

class controller(object):

    def __init__(self, tnet, vnet, nproc=4):

        # throttle network
        self.tnet = tnet
        # vector network
        self.vnet = vnet

    def train(self, data, epo=50, nb=10, lr=1e-4, ptst=0.1, nproc=4):

        # worker pool
        pool = Pool(nproc)

        # train in parallel
        self.tnet, self.vnet = pool.map(
            lambda conf: conf[0].train(data.i, conf[1], epo, nb, lr, ptst),
            [(self.tnet, data.o[:, 0]), (self.vnet, data.o[:, 1:4])]
        )

        pool.terminate()

        return self.tnet, self.vnet

    def __call__(self, x):

        # throttle
        throttle = self.tnet(x).data.numpy()

        # vector
        vector = self.vnet(x).data.numpy()

        return np.hstack((throttle, vector))

    def plot_prediction(self, data, marks=("kx", "k.")):

        # predicted output
        p = self.__call__(data.i)

        # actual output
        a = data.o.numpy()

        # throttle
        up, u = p[:, 0], a[:, 0]

        # vector
        uxp, uyp, uzp = p[:, 1], p[:, 2], p[:, 3]
        ux, uy, uz = a[:, 1], a[:, 2], a[:, 3]

        # marks
        pm = marks[0]
        am = marks[1]

        # plots
        f, ax = plt.subplots(4, sharex=True)
        ax[0].plot(up, pm)
        ax[0].plot(u, am)
        ax[0].set_ylabel("$u$")
        ax[1].plot(uxp, pm)
        ax[1].plot(ux, am)
        ax[1].set_ylabel("$u_x$")
        ax[2].plot(uyp, pm)
        ax[2].plot(uy, am)
        ax[2].set_ylabel("$u_y$")
        ax[3].plot(uzp, pm)
        ax[3].plot(uz, am)
        ax[3].set_ylabel("$u_z$")

        plt.tight_layout()
        plt.show()

    def plot_training(self):

        f, ax = plt.subplots(2, sharex=True)
        ax[0].plot(self.tnet.ltrn, "k-")
        ax[0].plot(self.tnet.ltst, "k--")
        #ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_ylabel("MSE")
        ax[0].set_title("Throttle Network")
        ax[0].legend(["Training Error", "Testing Error"])
        ax[1].plot(self.vnet.ltrn, "k-")
        ax[1].plot(self.vnet.ltst, "k--")
        #ax[1].set_xscale("log")
        ax[1].set_yscale("log")
        ax[1].set_ylabel("MSE")
        ax[1].set_xlabel("Epoch")
        ax[1].set_title("Vector Network")
        ax[1].legend(["Training Error", "Testing Error"])

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import os, numpy as np, cloudpickle as cp
    from mlp import sphere
    from data import data
    __dir__ = os.path.dirname(os.path.realpath(__file__))

    throttle = cp.load(open(__dir__ + "/../p/throttle.p", "rb"))
    vector = cp.load(open(__dir__ + "/../p/vector.p", "rb"))
    cont = controller(throttle, vector)

    # get trajectory data
    traj = np.load(__dir__ + "/../npy/moc_data.npy")[7000:7200]
    # ordered data
    dbo = data(traj, [1,2,3,4,5,6,7], [16,17,18,19], shuffle=False)

    cont.plot_training()
    cont.plot_prediction(dbo, marks=("k--", "k-"))
