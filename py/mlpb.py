import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocess as mp
import matplotlib.pyplot as plt

class mlp(torch.nn.Sequential):

    def __init__(self, shape, drop=0.2, name="Base"):

        # architecture
        self.shape = shape

        # number of inputs
        self.ni = self.shape[0]

        # number of outputs
        self.no = self.shape[-1]

        # number of layers
        self.nl = len(self.shape)

        # scaler
        self.scaler = StandardScaler()

        # loss data
        self.ltrn, self.ltst = list(), list()

        # name
        self.name = name

        # operations
        self.ops = list()

        # apply operations
        for i in range(self.nl - 1):

            # linear layer
            self.ops.append(torch.nn.Linear(self.shape[i], self.shape[i + 1]))

            # if penultimate layer
            if i == self.nl - 2:

                # output between 0 and 1
                self.ops.append(torch.nn.Sigmoid())
                #pass

            # if hidden layer
            else:

                # dropout
                self.ops.append(torch.nn.Dropout(p=drop))

                # activation
                self.ops.append(torch.nn.ReLU())

        # initialise neural network
        torch.nn.Sequential.__init__(self, *self.ops)
        self.double()

    @staticmethod
    def format(x):
        if isinstance(x, np.ndarray):
            x = torch.autograd.Variable(torch.from_numpy(x)).double()
        elif isinstance(x, torch.DoubleTensor):
            x = torch.autograd.Variable(x).double()
        elif isinstance(x, torch.autograd.Variable):
            x = x.double()
        else:
            raise TypeError("Det finns et problem.")
        return x

    def forward(self, x):

        # format input
        x = self.format(x)

        # apply operations
        for op in self.ops:
            x = op(x)

        # return transformation
        return x

    def train(self, idat, odat, epo=50, batches=10, lr=1e-4, ptst=0.1):

        # numpy
        idat, odat = [dat.numpy() for dat in [idat, odat]]

        # fit scaler
        self.scaler.fit(idat)

        # transform inputs
        idat = self.scaler.transform(idat)

        # number of testing samples
        ntst = int(ptst*idat.shape[0])

        # training data
        itrn, otrn = [dat[ntst:] for dat in [idat, odat]]

        # testing data
        itst, otst = [dat[:ntst] for dat in [idat, odat]]

        # delete original data
        del idat; del odat

        # batch data
        itrn, otrn, itst, otst = [np.array_split(dat, batches) for dat in [itrn, otrn, itst, otst]]

        # tensor
        itrn, otrn, itst, otst = [[torch.from_numpy(d) for d in dat] for dat in [itrn, otrn, itst, otst]]

        # variables
        itrn, otrn, itst, otst = [[torch.autograd.Variable(d) for d in dat] for dat in [itrn, otrn, itst, otst]]

        # optimiser
        opt = torch.optim.SGD(self.parameters(), lr=lr)

        # loss function
        lf = torch.nn.MSELoss()

        # for each episode
        for t in range(epo):

            # average episode loss
            ltrne, ltste = list(), list()

            # zero gradients
            opt.zero_grad()

            # for each batch
            for itrnb, otrnb, itstb, otstb in zip(itrn, otrn, itst, otst):

                # training loss
                ltrn = lf(self(itrnb), otrnb)

                # testing loss
                ltst = lf(self(itstb), otstb)

                # record loss
                ltrne.append(float(ltrn.data[0]))
                ltste.append(float(ltst.data[0]))

                # progress
                print(self.name, t, ltrn.data[0], ltst.data[0])

                # backpropagate training error
                ltrn.backward()

            # update weights
            opt.step()

            self.ltrn.append(np.average(ltrne))
            self.ltst.append(np.average(ltste))

        return self


    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.ltrn)
        ax.plot(self.ltst)
        return ax


class cartesian(mlp):

    def __init__(self, hshape, ni=7, drop=0.2):

        # base network
        mlp.__init__(self, [ni] + hshape + [3], drop, "Cartesian")

    def forward(self, x):

        # base network computation
        x = mlp.forward(self, x)

        # 1) signals
        s1, s2, s3 = x[:, 0], x[:, 1], x[:, 2]

        # 2) spherical parametres
        rho, theta, phi = s1, s2*np.pi, s3*2*np.pi

        # 3) Cartesian components
        st = torch.sin(theta)
        ct = torch.cos(theta)
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        u, ux, uy, uz = rho, st*cp, st*cp, ct

        return torch.stack((u, ux, uy, uz), dim=1)



class sphere(mlp):

    def __init__(self, shape, drop=0.2):
        self.t1 = torch.autograd.Variable(torch.Tensor([1, 2*np.pi, np.pi])).double()
        self.t2 = torch.autograd.Variable(torch.Tensor([0, -np.pi, 0])).double()
        mlp.__init__(self, shape, drop, "Spherical")

    def forward(self, x):
        return mlp.forward(self, x).mul(self.t1).add(self.t2)

class throttle(mlp):

    def __init__(self, hshape, ni=7, drop=0.2):
        mlp.__init__(self, [ni] + hshape + [1], drop, "Throttle")

class azimuthal(mlp):

    def __init__(self, shape, drop=0.2):
        mlp.__init__(self, shape, drop, "Azimuthal")

    def forward(self, x):
        return mlp.forward(self, x).mul(2*np.pi).add(-np.pi)

class polar(mlp):

    def __init__(self, shape, drop=0.2):
        mlp.__init__(self, shape, drop, "Polar")

    def forward(self, x):
        return mlp.forward(self, x).mul(np.pi)

class parametrised(mlp):

    def __init__(self, shape, drop=0.2):
        mlp.__init__(self, shape, drop, "Parametrised")

    def forward(self, x):
        x = mlp.forward(self, x)

        # 1) signal {[0,1], [0,1], [0, 1]}
        o1, o2, o3 = x[:, 0], x[:, 1], x[:, 2]

        # 2) parametres {[0,1], [0, 2pi], [-1, 1]}
        u, theta, z = o1, o2*2*np.pi, o3*2 - 1

        # 3) physical
        p  = torch.sqrt(1 - z**2)
        ux = p*torch.cos(theta)
        uy = p*torch.sin(theta)
        uz = z

        return torch.stack((u, ux, uy, uz), dim=1)
