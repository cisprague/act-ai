import os, numpy as np, cloudpickle as cp
from mlp import sphere
from data import data
__dir__ = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    # get trajectory data
    traj = np.load(__dir__ + "/../npy/moc_data.npy")[5000:5200]
    # ordered data
    dbo = data(traj, [1,2,3,4,5,6,7], [17, 18, 19], shuffle=False)
    # random data
    dbr = data(traj, [1,2,3,4,5,6,7], [17, 18, 19], shuffle=True)

    # check for previous network
    try:
        net = cp.load(open(__dir__ + "/../p/vector.p", "rb"))
        print("Found prexisting vector network of shape " + str(net.shape))
    # otherwise create one
    except: net = sphere([100]*5)

    # train neural network
    while True:
        net.train(dbr.i, dbr.o, epo=10000, batches=1, lr=1e-8, ptst=0.01)
        cp.dump(net, open(__dir__ + "/../p/vector.p", "wb"))
