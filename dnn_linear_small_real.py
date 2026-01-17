import numpy as np

from model.lib_main_nn import (
    Activation_Softmax,
    Activation_Softmax_Loss_CCEntropy,
    Layer_Dense,
)
from model.optimizers.Adagrad import Adagrad_Optimizer
from model.optimizers.Adam import Adam_Optimizer
from model.optimizers.RMSProp import RMSProp_Optimizer
from model.optimizers.SGD import SGD_Optimizer

# Model expects samples and features, not singals
# y_pred = (samples, classes)
# y_true = (samples,) or y_true(samples, classes)
# Generally in classification X.shape[0] == y_labels.shape[0]
# Where X is the matrix of samples
# low–moderate SNR.
# goal is for the network to learn the mapping (H,y) -> x

# current model limitations
# 1. only reald
# 2. trained on a single SNR value
# 3. doesn't use a match filter approach (batch)
# 4. is linear such that it's comparable to ZF
# 5. not large enough for mimo large, inherent structur of the layers and neurons must change

# The success of this model is that it learned y = Hx + noise
# which was
# . 1. all values are real
#  2. element wise multiplication, no MF
#  3. single noise value

Nr = 2
Nt = 2
M = 2
K = int(1e3)
SNR = np.arange(0, 22, 2)
num_combinations = M**Nt

angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(angles)

# z = np.zeros((Nt, num_combinations), dtype=int)
# z = np.array(np.unravel_index(np.arange(M**Nt), (M,) * Nt))
# combinML = PSK_Table[z]

H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K))

dataIn = np.random.randint(0, M, (Nt, K))
dataMod = PSK_Table[dataIn]
x = dataMod

y = np.einsum("rnk,nk->rk", H, x)

snr_db = 10
snr = 10 ** (snr_db / 10)
sigma = np.sqrt(1 / (2 * snr))
noise = sigma * np.random.randn(*y.shape)
y_noise = y + noise

y_samples = y_noise.T
H_samples = np.transpose(H, (2, 0, 1)).reshape(K, Nr * Nt)
X = np.hstack([y_samples, H_samples])
y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)

# -----------------DNN--------------------

# try learning rate 1e-4 or 3e-4, log explosion
# current best learning trainng rate 0.078
layer1 = Layer_Dense(6, 32)
act1 = Activation_Softmax()
layer2 = Layer_Dense(32, num_combinations)
loss_func = Activation_Softmax_Loss_CCEntropy()
optimizer1 = Adam_Optimizer(learning_rate=1e-3, decay=1e-2, beta1=0.99, beta2=0.999)
optimizer2 = Adagrad_Optimizer(learning_rate=0.7, decay=1e-6)
optimizer3 = RMSProp_Optimizer(learning_rate=0.05, decay=1e-2, rho=0.7)
optimizer4 = SGD_Optimizer(learning_rate=1, decay=1e-6, momentum=0.9)

# Each Epoch is a single forwrad then backward pass
for epoch in range(1001):
    # Single forward pass
    layer1.forward(X)
    act1.forward(layer1.output)
    layer2.forward(act1.output)

    # Calcuating Loss, prediction, accuracy
    loss = loss_func.forward(layer2.output, y_labels)

    predictions = np.argmax(loss_func.output, axis=1)

    if len(y_labels.shape) == 2:
        y_labels = np.argmax(y_labels, axis=1)
    accuracy = np.mean(predictions == y_labels)

    # single backward pass, calculating the dinputs of each layer
    # i.e. calculating the deltas of weights and biases of each layer
    loss_func.backward(loss_func.output, y_labels)
    layer2.backward(loss_func.dinputs)
    act1.backward(layer2.dinputs)
    layer1.backward(act1.dinputs)

    # Now finally updating the weights and biases
    optimizer1.pre_update_params()
    optimizer1.update_params(layer1)
    optimizer1.update_params(layer2)
    optimizer1.post_update_params()

    if not epoch % 100:
        print(
            f"Epoch: {epoch}, lr: {optimizer1.current_learning_rate}, Loss: {
                loss:.3f}, Acc: {accuracy:.3f}"
        )


# --------------END OF DNN-----------------------------


# ---------------ML Detection--------------------------
# -------------------Testing----------------------------


angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(angles)

H_test = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K))

dataIn_test = np.random.randint(0, M, (Nt, K))
dataMod_test = PSK_Table[dataIn_test]
x_test = dataMod_test

y_test = np.einsum("rnk,nk->rk", H_test, x_test)

snr_db = 10
snr = 10 ** (snr_db / 10)
sigma = np.sqrt(1 / (2 * snr))
noise = sigma * np.random.randn(*y.shape)
y_noise = y_test + noise

y_test_samples = y_noise.T
H_test_samples = np.transpose(H_test, (2, 0, 1)).reshape(K, Nr * Nt)
X_test = np.hstack([y_test_samples, H_test_samples])
y_test_labels = np.ravel_multi_index(dataIn_test, (M,) * Nt)

layer1.forward(X_test)
act1.forward(layer1.output)
layer2.forward(act1.output)

loss = loss_func.forward(layer2.output, y_test_labels)
preds = np.argmax(loss_func.output, axis=1)

if len(y_test_labels.shape) == 2:
    y_test_labels = np.argmax(y_test_labels, axis=1)

acc = np.mean(preds == y_test_labels)

print("\n-------testing--------------\n", f"loss= {loss}, acc= {acc}")
