import numpy as np
import matplotlib.pyplot as plt
import os
import cloudpickle as cp
import torch
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from mlp import mlp
from types import MethodType
import multiprocess as mp

class controller(object):

    def __init__(self, net):

        # unified neural network
        if isinstance(net, mlp) and net.ni == 7 and net.no == 3:

            # assign network
            self.net = net

            # transformations
            t1 = torch.autograd.Variable(torch.Tensor([1, 2*np.pi, np.pi]), requires_grad=False).double()
            t2 = torch.autograd.Variable(torch.Tensor([0, -np.pi, 0]), requires_grad=False).double()

            # transform output
            self.net.forward = MethodType(
                lambda net, x: net._operate(x).mul(t1).add(t2),
                self.net
            )

            # forward pass of network
            self.forward = MethodType(
                lambda self, x: self.net(x),
                self
            )

        # dedicated neural networks
        elif hasattr(net, '__iter__') and len(net) == 3 and all([n.ni == 7 and n.no == 1 for n in net]):

            # assign networks
            self.nets = net

            # throttle network
            self.nets[0].forward = MethodType(
                lambda net, x: self.nets[0]._operate(x),
                self.nets[0]
            )

            # azimuth network
            self.nets[1].forward = MethodType(
                lambda net, x: self.nets[1]._operate(x).mul(2*np.pi).add(-np.pi),
                self.nets[1]
            )

            # polar network
            self.nets[2].forward = MethodType(
                lambda net, x: self.nets[2]._operate(x).mul(np.pi),
                self.nets[2]
            )

            # forwards pass of networks
            self.forward = MethodType(
                lambda self, x: torch.cat([net(x) for net in self.nets], dim=1),
                self
            )

        # error
        else:
            raise TypeError("Det finns ett problem.")


    def __call__(self, x):

        if isinstance(x, np.ndarray):
            x = torch.autograd.Variable(torch.from_numpy(x)).double()
        elif isinstance(x, torch.DoubleTensor):
            x = torch.autograd.Variable(x).double()
        elif isinstance(x, torch.autograd.Variable):
            x = x.double()
        else:
            raise TypeError("Det finns et problem.")

        return self.forward(x)

    def train(self, idat, odat, epo=50, lr=1e-4, ptst=0.1, batches=10):

        # unified network
        if hasattr(self, "net"):
            self.net =  self.net.train(idat, odat, epo=epo, lr=lr, ptst=ptst, batches=batches)

        # dedicated networks
        elif hasattr(self, "nets"):

            # dedicated workers
            p = mp.Pool(3)

            # train in parallel
            self.nets = p.map(lambda i: self.nets[i].train(idat, odat[:, i], epo=epo, lr=lr, ptst=ptst, batches=batches), range(3))
