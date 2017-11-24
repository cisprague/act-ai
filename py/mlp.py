import torch, numpy as np
from sklearn.preprocessing import StandardScaler

class mlp(torch.nn.Module):

    def __init__(self, shape, drop=0, name="MLP"):

        super(mlp, self).__init__()

        self.shape  = shape
        self.ni     = self.shape[0]
        self.no     = self.shape[-1]
        self.nl     = len(self.shape)
        self.scaler = StandardScaler()
        self.ltrn   = list()
        self.ltst   = list()
        self.name   = name
        self.double()

        op = 0
        for i in range(self.nl -1):

            setattr(self, 'op' + str(op), torch.nn.Linear(self.shape[i], self.shape[i + 1]))
            op += 1

            if i == self.nl - 2:

                setattr(self, 'op' + str(op), torch.nn.Sigmoid())
                op += 1

            else:

                setattr(self, 'op' + str(op), torch.nn.Dropout(p=drop))
                op += 1

                setattr(self, 'op' + str(op), torch.nn.ReLU())
                op += 1

    def train(self, idat, odat, epo, lr, nb, pt):

        self.scaler.fit(idat)

        idat = self.scaler.transform(idat)

        ntst = int(pt*idat.shape[0])

        itrn, otrn = [dat[ntst:] for dat in [idat, odat]]

        itst, otst = [dat[:ntst] for dat in [idat, odat]]

        del idat; del odat

        itrn, otrn, itst, otst = [np.array_split(dat, nb) for dat in [itrn, otrn, itst, otst]]

        itrn, otrn, itst, otst = [[torch.autograd.Variable(torch.from_numpy(d)).double() for d in dat] for dat in [itrn, otrn, itst, otst]]

        opt = torch.optim.Adam(self.parameters(), lr=lr)

        lf = torch.nn.MSELoss(self.parameters())

        for t in range(epo):

            opt.zero_grad()

            for itrnb, otrnb, itstb, otstb in zip(itrn, otrn, itst, otst):

                ltrn = lf(self(itrnb), otrnb)

                ltst = lf(self(itstb), otstb)

                self.ltrn.append(float(ltrn.data[0]))
                self.ltst.append(float(ltst.data[0]))

                print(self.name, t, ltrn.data[0], ltst.data[0])

                ltrn.backward()

            opt.step()

        return self
