import numpy as np

# Generally in each class would want a forward and backward method
# Bug: in the dot prodcut, there is no way to set a default data type for it, so sometimes it chooses different kinds of data types for the array

np.random.seed(0)

# inputs = np.array(
#    [[-1.3, 4.4, 5.5, 4.4], [-3.1, 1.11, 0.27, 2.3], [2.12, 3.3, -6.7, 5.5]],
#    dtype=float,
# )


class Layer_Dense:
    """Define neural network"""

    def __init__(self, inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # Grandient on params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    """Defines ReLU activation function"""

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Make copy of dvalues to modify it
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    """Defines softmax activation function"""

    def forward(self, inputs):
        # Take the max of each array in the batch not max of entire batch
        # max takes an ndim array, with an axis
        # maximum, takes two arrays, flattens them and produces max val in one array
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        """Dense Layer Expects 2D array, must flatten softmax output"""
        # Create an uninit array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalue) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flatten out output array, to be 2d
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # calculate and add up sample-wise (row) gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)


class Loss:
    """General class to define loss functions"""

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_Cross_Categorical_Entropy(Loss):
    """Defines CCE as a loss function, inherits from Loss"""

    # index example : [0, 1]
    # index example: [[0,1], [0,1]]
    # require handle both using one-hot encoded

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = []

        if len(y_true.shape) == 1:
            # Eg: [0, 1, 0, 1] 1D array
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likleiehood = -np.log(correct_confidences)
        return negative_log_likleiehood

    def backward(self, dvalues, y_true):
        samples = len(y_true)
        labels = len(dvalues[0])  # the lenght of each label per sample

        # if the lables are sparse, one-hot encode them
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient and normalize
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CCEntropy:
    """Combined softmax and CCEntropy, it's faster than separate"""

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Cross_Categorical_Entropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # if lables are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1  # Gradient
        self.dinputs = self.dinputs / samples  # Normalize
