import numpy as np

def perturb(vector, percent):
    # returns a vector randomly perturbed by a percent
    n = len(vector)
    vector = np.asarray(vector)
    # random normal distribution. mean=0, var=1
    rand = np.random.randn(n)
    # perturb vector
    pert = vector*percent
    pert = np.multiply(pert, rand)
    return vector + pert



if __name__ == "__main__":

    # example vector
    r = np.array([100, 255, 678])

    print(perturb(r, 0.05))
