from mlp import *
from data import data
import os
import numpy as np
import cloudpickle as cp
import multiprocess as mp
import matplotlib.pyplot as plt
import matplotlib as mpl

# path of this file
fp = os.path.realpath(__file__)
fp = os.path.split(fp)[0]

# data path
fdat = fp + "/../npy/xu_moc_data_cart.npy"

# neural network path
fnets = fp + "/../p/nets.p"

# define network shape
shape = [7, 100, 100, 1]

# get data
rdata = np.load(fdat)

# define database
data = data(rdata, [0, 1, 2, 3, 4, 5, 6], [7, 8, 9])

# define the network
try:

    # attempt to load the network
    nets = cp.load(open(fnets, "rb"))

    # make sure it is desired shape
    for net in nets:
        if net.shape != shape:
            raise Exception("Matching net not found.")

except Exception as e:

    # print notification
    print(e)

    # instantiate new network is not found
    nets = [throttle(shape), azimuthal(shape), polar(shape)]

# set up workers
p = mp.Pool(3)

# train nets in parallel
nets = p.map(lambda i: nets[i].train(data.i, data.o[:, i], epo=5000, batches=10, ptst=0.1), range(3))

# save networks
cp.dump(nets, open(fnets, "wb"))

# plot
mpl.style.use('seaborn')
fig, axs = plt.subplots(2, 3)

for net, lax, dax, i, unit in zip(nets, axs[0], axs[1], [7, 8, 9], ["Throttle [ND]", "Azimuth [rad]", "Polar [rad]"]):

    # loss
    lax = net.plot(lax)
    lax.legend(["Training Loss", "Testing Loss"])
    lax.set_title(net.name + " Network " + str(net.shape))
    lax.set_xlabel("Epoch")
    lax.set_ylabel("MSE Loss")

    # physical
    n = 400
    dax.plot(rdata[:n, i], ".")
    dax.plot(net(rdata[:n, :7]).data.numpy(), ".")
    dax.set_ylabel(unit)
    dax.legend(["Truth", "Predicted"])

plt.show()
