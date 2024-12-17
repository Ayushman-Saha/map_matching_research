import numpy as np


class Parameter:
    def __init__(self, values, mean=None, std=None):
        self.values = values
        self.mean = mean or np.mean(values)
        self.std = std or np.std(values)

    def normalize(self):
        """
        Apply sigmoid normalization to the parameter values.
        """
        return [1 / (1 + np.exp(-(value - self.mean) / self.std)) for value in self.values]