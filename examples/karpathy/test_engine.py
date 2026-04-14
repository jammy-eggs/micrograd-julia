import math
import unittest

from engine import Value


class EngineTest(unittest.TestCase):
    def test_tanh_forward(self):
        self.assertEqual(Value(0.0).tanh().data, 0.0)
        self.assertAlmostEqual(Value(1.0).tanh().data, math.tanh(1.0))
        self.assertAlmostEqual(Value(-2.0).tanh().data, math.tanh(-2.0))

    def test_tanh_gradient(self):
        value = Value(0.5)
        out = value.tanh()
        out.grad = 1.0
        out._backward()
        self.assertAlmostEqual(value.grad, 1.0 - math.tanh(0.5) ** 2)

    def test_backward_handles_deep_graph_without_recursion_error(self):
        value = Value(1.0)
        out = value
        for _ in range(2000):
            out = out + 1.0
        out.backward()
        self.assertEqual(value.grad, 1.0)


if __name__ == "__main__":
    unittest.main()
