import numpy as np


class Activation:
    def __init__(self, thresholds):
        self._thresholds = thresholds

    def __call__(self, inputs):
        if inputs.shape[-1] != self._thresholds.shape[-1]:
            print("Thresholds and inputs shapes don't match,"
                  " auto filling with the first value")
            self._thresholds = np.ones(inputs.shape) * self._thresholds[0]

        outputs = np.zeros(inputs.shape)
        outputs[inputs >= self._thresholds] = 1

        return outputs


class DenseLayer:
    def __init__(self, weights):
        self._weights = weights

    def __call__(self, inputs):
        return np.matmul(inputs, self._weights.T)
