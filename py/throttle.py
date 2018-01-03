import os, numpy as np, cloudpickle as cp
from mlp import throttle
from data import data
__dir__ = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    # get trajectory data
    traj = np.load(__dir__ + "/../npy/moc_data.npy")[7000:7200]
    # ordered data
    dbo = data(traj, [1,2,3,4,5,6,7], 16, shuffle=False)

    # check for previous network
    try:
        net = cp.load(open(__dir__ + "/../p/throttle.p", "rb"))
        print("Found prexisting throttle network of shape " + str(net.shape))
    # otherwise create one
    except: net = throttle([100]*5)

    # train neural network
    while True:
        dbr = data(traj, [1,2,3,4,5,6,7], 16, shuffle=True)
        net.train(dbr.i, dbr.o, epo=10000, batches=1, lr=1e-8, ptst=0.01)
        #net.train(dbr.i, dbr.o, epo=5000, batches=1, lr=1e-8, ptst=0.05)
        cp.dump(net, open(__dir__ + "/../p/throttle.p", "wb"))
