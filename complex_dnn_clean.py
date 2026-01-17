import matplotlib.pyplot as plt
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

Nr = 2
Nt = 2
M = 2
K = int(1e4)
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
X_mean = (X - mu) / sigma
X = X_mean

y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)

# -----------------DNN--------------------

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

    if not epoch % 50:
        print(
            f"Epoch: {epoch}, lr: {optimizer1.current_learning_rate}, Loss: {
                loss:.3f}, Acc: {accuracy:.3f}"
        )


# --------------END OF DNN-----------------------------


# -------------------Testing----------------------------

SNR = np.arange(0, 22, 2)
bit_number = Nt * K * np.log2(M)

angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(1j * angles)

num_combinations = M**Nt
z = np.zeros((Nt, num_combinations), dtype=int)
z = np.array(np.unravel_index(np.arange(M**Nt), (M,) * Nt))
combinML = PSK_Table[z]

H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))

dataIn = np.random.randint(0, M, (Nt, K))
dataMod = PSK_Table[dataIn]
x = dataMod

y = np.einsum("rnk,nk->rk", H, x)

power_rx = np.mean(np.abs(x) ** 2)
ser_NN = np.zeros(len(SNR))

for i in range(len(SNR)):
    power_n = SNR[i] - 10 * np.log10(power_rx)
    sigma = 10 ** (-power_n / 10)
    noise = (np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)) * np.sqrt(sigma / 2)
    y_noise = y + noise

    # ---------------NN input MF-----------------
    z = np.einsum("rnk,rk->nk", H.conj(), y_noise)

    X = np.hstack(
        [
            z.real.T,
            z.imag.T,
            H.transpose(2, 0, 1).reshape(K, -1).real,
            H.transpose(2, 0, 1).reshape(K, -1).imag,
        ]
    )

    # Normalize using training mean
    # X = (X - X_mean) / (X.std(axis=0) + 1e-8)

    # Normalize using current testing mean
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_mean = (X - mu) / sigma

    # layer2.output.shape = (K, Nr * Nt)
    layer1.forward(X)
    act1.forward(layer1.output)
    layer2.forward(act1.output)

    pred_joint = np.argmax(layer2.output, axis=1)
    pred_symbols = np.array(np.unravel_index(pred_joint, (M,) * Nt))

    ser_NN[i] = np.sum(pred_symbols != dataIn) / (Nt * K)


plt.figure()

plt.semilogy(SNR, ser_NN, "s-", label="NN")

plt.grid(True)

plt.xlabel("SNR [dB]")
plt.ylabel("SER")
plt.title("2x2 MIMO SER Comparison")
plt.legend()

plt.show()
