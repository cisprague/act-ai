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
from mlp import *
from data import *

class controller(object):

    def __init__(self):

        # transformations
        self.t1 = torch.autograd.Variable(torch.Tensor([1, 2*np.pi, np.pi])).double()
        self.t2 = torch.autograd.Variable(torch.Tensor([0, -np.pi, 0])).double()


class unified(controller):

    def __init__(self, net):

        controller.__init__(self)

        self.net = net

        self.net.forward = MethodType(
            lambda net, x: net.operate(x).mul(self.t1).add(self.t2),
            self.net
        )

class dedicated(controller):

    def __init__(self, nets):

        controller.__init__(self)

        self.nets = nets

        for net, m, a in zip(self.nets, self.t1, self.t2):

            net.forward = MethodType(
                lambda nn, x: nn.operate(x).mul(m).add(a),
                net
            )
