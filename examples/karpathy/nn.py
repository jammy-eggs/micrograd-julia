import random

from engine import Value

ACTIVATIONS = {"relu", "tanh"}


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True, activation="relu"):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation: {activation}")
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        if not self.nonlin:
            return act
        return act.relu() if self.activation == "relu" else act.tanh()

    def parameters(self):
        return [*self.w, self.b]


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activation="relu"):
        sz = [nin, *nouts]
        self.layers = [
            Layer(
                sz[i],
                sz[i + 1],
                nonlin=i != len(nouts) - 1,
                activation=activation,
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
