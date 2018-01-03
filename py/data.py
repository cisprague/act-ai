import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class data(object):

    def __init__(self, data, cin, cout, shuffle=True):

        # cast as numpy array
        data = np.array(data)

        # shuffle rows
        if shuffle:
            np.random.shuffle(data)

        # number of samples
        self.n = data.shape[0]

        # cast to torch
        self.i = torch.from_numpy(data[:, cin]).double()
        self.o = torch.from_numpy(data[:, cout]).double()
