import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocess as mp

class mlp(torch.nn.Sequential):

    def __init__(self, shape=[7, 10, 10, 1], drop=0.2, acth=torch.nn.ReLU):

        # architecture
        self.shape = shape

        # number of features
        self.ni = self.shape[0]
        self.no = self.shape[-1]

        # number of layers
        self.nl = len(self.shape)

        # operations
        self.ops = list()

        # for each layer except last
        for i in range(self.nl - 1):

            # affine linear transformation
            self.ops.append(torch.nn.Linear(self.shape[i], self.shape[i + 1]))

            # if penultimate layer
            if i == self.nl - 2:

                # outputs between 0 an 1
                self.ops.append(torch.nn.Sigmoid())


            # if hidden layer
            else:

                # dropout
                self.ops.append(torch.nn.Dropout(p=drop))

                # rectified activation
                self.ops.append(acth())

        # initialise the nerual network
        torch.nn.Sequential.__init__(self, *self.ops)
        self.double()

        # loss records
        self.ltrn = list()
        self.ltst = list()

    def _operate(self, x):

        # apply operations
        for op in self.ops:
            x = op(x)

        return x

    def train(self, idat, odat, epo=50, batches=10, lr=1e-4, ptst=0.1):

        # numpy
        idat, odat = [dat.numpy() for dat in [idat, odat]]

        # set scaler
        self.scaler = StandardScaler().fit(idat)

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

            # for each batch
            for itrnb, otrnb, itstb, otstb in zip(itrn, otrn, itst, otst):

                # zero gradients
                opt.zero_grad()

                # training loss
                ltrn = lf(self(itrnb), otrnb)

                # testing loss
                ltst = lf(self(itstb), otstb)

                # record loss
                ltrne.append(float(ltrn.data[0]))
                ltste.append(float(ltst.data[0]))

                # progress
                print(t, ltrn.data[0], ltst.data[0])

                # backpropagate training error
                ltrn.backward()

                # update weights
                opt.step()

            self.ltrn.append(np.average(ltrne))
            self.ltst.append(np.average(ltste))

        return self
