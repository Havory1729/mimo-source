import matplotlib.pyplot as plt
import numpy as np

from data_set.data_set import create_mimo_data
from model.lib_main_nn import *
from model.optimizers.RMSProp import RMSProp_Optimizer

# For hidden layers use ReLU
# Because softmax is an output layer act func
# in the hidden layers it hinders data representation
# Previous problems, single channel detection
# no SNR awareness to beat MMSE


"""
current problems
1. need better SNR range
2. use gram method to beat MMSE
3. need better physics implementation for non linearity to simulate MMSE equation 

"""

Nr = 2
Nt = 2
M = 2
K = int(1e4)
SNR = np.arange(0, 22, 2)
num_combinations = M**Nt

H, X, X_train_mean, X_train_std, x, y, y_labels, datain = create_mimo_data(
    Nr, Nt, M, K, SNR, num_combinations
)

# -----------------DNN--------------------

layer1 = Layer_Dense(13, 16)
act1 = Activation_ReLU()
layer2 = Layer_Dense(16, num_combinations)
loss_func = Activation_Softmax_Loss_CCEntropy()
optimizer3 = RMSProp_Optimizer(learning_rate=0.005, decay=1e-4, rho=0.9)

# Each Epoch is a single forwrad & backward pass
for epoch in range(1001):
    snr_db = np.random.uniform(0, 20)
    snr_feature = np.full((K, 1), snr_db / 20.0)
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    noise = sigma * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))

    H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))
    y_clean = np.einsum("rnk,nk->rk", H, x)
    y_noise = y_clean + noise

    z = np.einsum("rnk,rk->nk", H.conjugate(), y_noise)

    X = np.hstack(
        [
            z.real.T,
            z.imag.T,
            H.transpose(2, 0, 1).reshape(K, -1).real,
            H.transpose(2, 0, 1).reshape(K, -1).imag,
            snr_feature,
        ]
    )

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
    optimizer3.pre_update_params()
    optimizer3.update_params(layer1)
    optimizer3.update_params(layer2)
    optimizer3.post_update_params()

    if not epoch % 10:
        print(
            f"Epoch: {epoch}, lr: {optimizer3.current_learning_rate:.3f}, Loss: {
                loss:.3f}, Acc: {accuracy:.3f}"
        )


# --------------END OF DNN-----------------------------


# -------------------Testing----------------------------


H, X, xmean, xstd, x, y, y_labels, dataIn = create_mimo_data(
    Nr, Nt, M, K, SNR, num_combinations
)

power_rx = np.mean(np.abs(x) ** 2)
ser_adam = np.zeros(len(SNR))

print_loss = 0
print_accuracy = 0

layer1.forward(X)
act1.forward(layer1.output)
layer2.forward(act1.output)

loss = loss_func.forward(layer2.output, y_labels)
preds = np.argmax(layer2.output, axis=1)
accruacy = np.mean(preds == y_labels)

# print("\n--------------RMSProp Testing-------------------\n")
# print(f"loss= {loss}, acc= {accuracy}")

for i in range(len(SNR)):
    power_n = SNR[i] - 10 * np.log10(power_rx)
    snr_feature = np.full((K, 1), power_n / 20.0)
    sigma = 10 ** (-power_n / 10)
    noise = (np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)) * np.sqrt(sigma / 2)

    H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))

    y_clean = np.einsum("rnk,nk->rk", H, x)
    y_noise = y_clean + noise

    # ---------------NN input MF-----------------
    z = np.einsum("rnk,rk->nk", H.conj(), y_noise)

    X = np.hstack(
        [
            z.real.T,
            z.imag.T,
            H.transpose(2, 0, 1).reshape(K, -1).real,
            H.transpose(2, 0, 1).reshape(K, -1).imag,
            snr_feature,
        ]
    )

    # layer2.output.shape = (K, Nr * Nt)
    layer1.forward(X)
    act1.forward(layer1.output)
    layer2.forward(act1.output)

    pred_joint = np.argmax(layer2.output, axis=1)
    pred_symbols = np.array(np.unravel_index(pred_joint, (M,) * Nt))

    ser_adam[i] = np.sum(pred_symbols != dataIn) / (Nt * K)


# ----------------------Usual Mimo--------------------------

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

ser_ZF = np.zeros(len(SNR))
ser_MMSE = np.zeros(len(SNR))
ser_ML = np.zeros(len(SNR))

printout = None

# Simulate detection with different SNR values
for i in range(len(SNR)):
    power_n = SNR[i] - 10 * np.log10(power_rx)
    sigma = 10 ** (-power_n / 10)  # Noise variance
    noise = (np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)) * np.sqrt(sigma / 2)
    y_noise = y + noise

    # ZF & MMSE
    x_ZF = np.zeros((Nt, K), dtype=complex)
    x_MMSE = np.zeros((Nt, K), dtype=complex)
    I_Nt = np.eye(Nt)  # Identity matrix

    for j in range(K):
        Hj = H[:, :, j]  # Nr * Nt
        Gram = Hj.conj().T @ Hj  # Nr * Nt
        MF = Hj.conj().T @ y_noise[:, j]  # Nt * 1

        # ZF: x = (H'H)^(-1) H'y
        x_ZF[:, j] = np.linalg.solve(Gram, MF)

        # MMSE: x = (H'H + sigma^2 I)^(-1) H'y
        x_MMSE[:, j] = np.linalg.solve(Gram + sigma * I_Nt, MF)

    dataOut_ZF = np.argmin(
        np.abs(x_ZF[:, :, None] - PSK_Table[None, None, :]) ** 2, axis=2
    )
    dataOut_MMSE = np.argmin(
        np.abs(x_MMSE[:, :, None] - PSK_Table[None, None, :]) ** 2, axis=2
    )

    printout = np.argmin(
        np.abs(x_MMSE[:, :, None] - PSK_Table[None, None, :]) ** 2, axis=2
    )

    # Signal Error Rate
    ser_ZF[i] = np.sum(dataOut_ZF != dataIn) / (Nt * K)
    ser_MMSE[i] = np.sum(dataOut_MMSE != dataIn) / (Nt * K)

    # ML Detection
    x_ML = np.zeros((Nt, K), dtype=complex)

    for j in range(K):
        Hj = H[:, :, j]  # Nr * Nt

        diff = y_noise[:, j : j + 1] - Hj @ combinML
        dist = np.sum(np.abs(diff) ** 2, axis=0)

        idx = np.argmin(dist)
        x_ML[:, j] = combinML[:, idx]

    # Demodulate ML
    dataOut_ML = np.argmin(
        np.abs(x_ML[:, :, None] - PSK_Table[None, None, :]) ** 2, axis=2
    )
    ser_ML[i] = np.sum(dataOut_ML != dataIn) / (Nt * K)


plt.figure()

# semilogy plots (logarithmic y-axis)
plt.semilogy(SNR, ser_adam, "*--", label="RMSProp")
plt.semilogy(SNR, ser_MMSE, "o--", label="MMSE")
plt.semilogy(SNR, ser_ZF, "+--", label="ZF")
plt.semilogy(SNR, ser_ML, "--.", label="ML")

plt.grid(True)
plt.xlabel("SNR [dB]")
plt.ylabel("SER")
plt.title("2x2 MIMO SER, 10k Symbols, BPSK, 16 Neurons, 2 Layers")
plt.legend()

plt.show()
