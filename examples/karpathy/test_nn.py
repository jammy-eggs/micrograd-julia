import unittest

from engine import Value
from nn import MLP, Neuron


class NNTest(unittest.TestCase):
    def test_neuron_tanh_activation(self):
        neuron = Neuron(2, activation="tanh")
        self.assertEqual(neuron.activation, "tanh")
        out = neuron([Value(1.0), Value(1.0)])
        self.assertIsInstance(out, Value)
        self.assertLessEqual(out.data, 1.0)
        self.assertGreaterEqual(out.data, -1.0)

    def test_mlp_configurable_activation(self):
        model = MLP(3, [4, 1], activation="tanh")
        self.assertEqual(model.layers[0].neurons[0].activation, "tanh")
        self.assertFalse(model.layers[-1].neurons[0].nonlin)


if __name__ == "__main__":
    unittest.main()
