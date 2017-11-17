import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class database(object):

    def __init__(self, data, cin, cout):

        # cast as numpy array
        data = np.array(data)

        # shuffle rows
        np.random.shuffle(data)

        # number of samples
        self.ndat = data.shape[0]

        # cast to torch
        self.idat = torch.from_numpy(data[:, cin])
        self.odat = torch.from_numpy(data[:, cout])
