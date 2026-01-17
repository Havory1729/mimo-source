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

# In training now model memorizations happens early

# in a single testing run the prediction mistake for loss is very high
#  even if accuracy is very high


# 100% accuracy is not surprising in this case for the following reasons
# 1. small constellasion BPSK
# 2. Small mimo
# 3. MF
# 4. narrow SNR range
# 5. linear and deterministic physical equation

# It's a classic ML Bayes Rule classifier which is expected to be smooth and
# reach 100% accuracy in this way
# in this case overfitting would need to show much more severe symptoms

# critical insight: too much training can be harmful for good testing results
# in fact too much training makes the model over confident and can very well
# increase test loss and decrease test accuracy

Nr = 2
Nt = 2
M = 2
K = int(1e3)
SNR = np.arange(0, 22, 2)
num_combinations = M**Nt

angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(1j * angles)

H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))

dataIn = np.random.randint(0, M, (Nt, K))
dataMod = PSK_Table[dataIn]
x = dataMod

y = np.einsum("rnk,nk->rk", H, x)

snr_db = 10
snr = 10 ** (snr_db / 10)
sigma = np.sqrt(1 / (2 * snr))
noise = sigma * np.random.randn(*y.shape)
y = y + noise

# Generating samples features batch wise
H_batched = np.transpose(H, (2, 0, 1))  # (K, Nr, Nt)
y_batched = y.T  # (K, Nr)
Hh = H_batched.conj().transpose(0, 2, 1)  # (K, Nt, Nr)
y_col = y_batched[..., None]  # (K, Nr, 1)
z = Hh @ y_col  # (K, Nt, 1)
z = z.squeeze(-1)  # removes last dimension

X = np.hstack([z.real, z.imag, H.real.reshape(K, -1), H.imag.reshape(K, -1)])

mu = X.mean(axis=0)
sigma = X.std(axis=0) + 1e-8
X = (X - mu) / sigma

y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)


# latest testing result
# loss= 0.3341769371479269, acc= 0.929
# loss= 0.1735606359142625, acc= 0.944

# -----------------DNN--------------------

# try learning rate 1e-4 or 3e-4, log explosion
# current best learning trainng rate 0.078
layer1 = Layer_Dense(12, 16)
act1 = Activation_Softmax()
layer2 = Layer_Dense(16, num_combinations)
loss_func = Activation_Softmax_Loss_CCEntropy()
optimizer1 = Adam_Optimizer(learning_rate=1e-3, decay=1e-2, beta1=0.99, beta2=0.999)
optimizer2 = Adagrad_Optimizer(learning_rate=0.7, decay=1e-6)
optimizer3 = RMSProp_Optimizer(learning_rate=0.05, decay=1e-2, rho=0.7)
optimizer4 = SGD_Optimizer(learning_rate=1, decay=1e-6, momentum=0.9)

# Each Epoch is a single forwrad & backward pass
for epoch in range(3001):
    snr_db = np.random.uniform(0, 20)
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * np.random.randn(*y.shape)
    y = y + noise

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

    # single bkwd pass, calculating (dinputs, dweights, dbiases)
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


Nr = 2
Nt = 2
M = 2
K = int(1e3)

angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(1j * angles)

H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))

dataIn = np.random.randint(0, M, (Nt, K))
dataMod = PSK_Table[dataIn]
x = dataMod

y = np.einsum("rnk,nk->rk", H, x)

# Generating samples features batch wise
H_batched = np.transpose(H, (2, 0, 1))  # (K, Nr, Nt)
y_batched = y.T  # (K, Nr)
Hh = H_batched.conj().transpose(0, 2, 1)  # (K, Nt, Nr)
y_col = y_batched[..., None]  # (K, Nr, 1)
z = Hh @ y_col  # (K, Nt, 1)
z = z.squeeze(-1)  # removes last dimension

X = np.hstack([z.real, z.imag, H.real.reshape(K, -1), H.imag.reshape(K, -1)])

mu = X.mean(axis=0)
sigma = X.std(axis=0) + 1e-8
X = (X - mu) / sigma

y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)


layer1.forward(X)
act1.forward(layer1.output)
layer2.forward(act1.output)

# Calcuating Loss, prediction, accuracy
loss = loss_func.forward(layer2.output, y_labels)

predictions = np.argmax(loss_func.output, axis=1)

if len(y_labels.shape) == 2:
    y_labels = np.argmax(y_labels, axis=1)
accuracy = np.mean(predictions == y_labels)

print("\n-------------testing--------------------------\n")
print(f"loss= {loss}, acc= {accuracy}")
