from mlp import mlp

class throttle(mlp):

    def __init__(self, hshape, drop=0, name="Throttle", ni=7):

        mlp.__init__(self, [ni] + hshape + [1], drop, name)

    def forward(self, x):

        return mlp(x)
