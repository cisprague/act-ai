import numpy as np
import matplotlib.pyplot as plt
import os
import cloudpickle as cp
import torch
from sklearn.preprocessing import StandardScaler
import seaborn as sb

class controller(torch.nn.Module):

    def __init__(self, data, shape=[10, 15, 20, 10], perc=0.3):

        # shuffle data
        #np.random.shuffle(data)

        # number of samples
        self.ndat = data.shape[0]

        # percent of data to use as for testin
        self.ptst = perc

        # number of training samples
        self.ntst = int(self.ptst*self.ndat)

        # full inputs and outputs
        self.idat = np.copy(data[:, 0:7])
        self.odat = np.copy(data[:, 7:11])

        # input & output training data
        self.itrn = np.copy(self.idat[:self.ntst, 0:7])
        self.otrn = np.copy(self.odat[:self.ntst, 7:11])

        # input & output test data
        self.itst = self.idat[self.ntst:, 0:7]
        self.otst = self.odat[self.ntst:, 7:11]

        # remove mean and scale to unit variance
        self.scaler = StandardScaler()
        self.scaler.fit(self.idat)

        # preprocessed inputs
        self.itrnt = torch.autograd.Variable(torch.Tensor(self.scaler.transform(self.itrn)))
        self.itstt = torch.autograd.Variable(torch.Tensor(self.scaler.transform(self.itst)))

        # neural network architecture
        self.shape = [7] + list(shape) + [3]
        self.nl    = len(self.shape)

        # operations
        self.ops = list()

        # for each layer, except output
        for i in range(self.nl - 1):

            # affine function
            op = [torch.nn.Linear(self.shape[i], self.shape[i + 1])]

            # if penultimate layer
            if i == self.nl - 2:
                op.append(torch.nn.Sigmoid())

            # if any other layer
            else:
                op.append(torch.nn.ReLU())
                op.append(torch.nn.Dropout(p=0.5))

            # store operation
            self.ops.append(op)

        # initialise network
        torch.nn.Module.__init__(self)

    def forward(self, x):

        # network operations
        for ops in self.ops:
            for op in ops:
                x = op(x)

        # u, theta, z
        x = torch.stack([
            x[:, 0],
            x[:, 1]*2*np.pi,
            x[:, 2]*2 - 1
        ], 1)

        # u, ux, uy, uz
        x = torch.stack([
            x[:, 0],
            torch.sqrt(1 - x[:, 2]**2)*torch.sin(x[:, 1]),
            torch.sqrt(1 - x[:, 2]**2)*torch.cos(x[:, 1]),
            x[:, 2]
        ], 1)

        return x

if __name__ == "__main__":

	# solution filename
	fp = os.path.realpath(__file__)
	fp = os.path.split(fp)[0]
	fn = fp + "/../npy/xu_data.npy"
	
	# data
	data = np.load(fn)
	
	# initialise controller
	controller = controller(data)
