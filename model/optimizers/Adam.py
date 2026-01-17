import numpy as np


class Adam_Optimizer:
    def __init__(
        self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta1=0.9, beta2=0.999
    ):
        # beta1 is momentum, beta2 = rho for cache
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        """Ran once before any param updates"""
        if self.decay:
            self.currenet_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradiens (used to warm up the model in intial steps)
        layer.weight_momentums = (
            self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases
        )

        # Get correct momentum (updating in the layer stages)
        # iteration = 0 at start
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta1 ** (self.iterations + 1)
        )

        # Update cache with square gradients
        layer.weight_cache = (
            self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2
        )
        layer.bias_cache = (
            self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2
        )

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta2 ** (self.iterations + 1)
        )

        # Vanila Adagrad parameter update + normalization
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        """Ran once after any update params"""
        self.iterations += 1
