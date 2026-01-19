import matplotlib.pyplot as plt
import numpy as np

from data_set.data_set import *
from model.lib_main_nn import *
from model.optimizers.RMSProp import *

# For hidden layers use ReLU
# Because softmax is an output layer act func
# in the hidden layers it hinders data representation
# Previous problems, single channel detection
# no SNR awareness to beat MMSE


"""
current problems with RMSProp on large mimo
1. exploding weights, biases, softmax activations 
    O(K, M^K) as opposed to classic mimo, O(Nt^3, K)

2. exploding one hot labeling even if sparse
3. RMSProp copies the current weights for cache, increasing size even more


sols:
1. need per stream classification (per sample)
    eg (output size = Nt × M = 4 × 32 = 128) predict each layer seprately

2. need a regression style detector
    output = complex ^ Nt (real + imag) 
    loss = MSE

3. iterative method / unfolded detection where:
    Each layer mimics one MMSE / gradient step
    No exponential output
    Matches classical detectors structurall


My current options are to lower the dimensions of the problem
"""


Nr = 16  # Num of receiver's antennas (base station)
Nt = 4  # Num of transmitters (user equipments)
M = 8  # Use M-qam modulation
K = int(1e4)  # Num of symbols transmitted per user
SNR = np.arange(0, 22, 2)
num_combinations = M**Nt

# -------------------------------DNN---------------------------

H, X, X_train_mean, X_train_std, x, y, y_labels, datain = create_mimo_data(
    Nr, Nt, M, K, SNR, num_combinations
)


layer1 = Layer_Dense(137, 16)
act1 = Activation_ReLU()
layer2 = Layer_Dense(16, num_combinations)
loss_func = Activation_Softmax_Loss_CCEntropy()
optimizer3 = RMSProp_Optimizer(learning_rate=0.005, decay=1e-4, rho=0.9)


# Each Epoch is a single forwrad & backward pass
for epoch in range(101):
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

# ----------------------Usual LS Mimo--------------------------


SNR = np.arange(5, 16)  # 5:15 inclusive
NS_order = 3
NI_order = 3
GS_order = 3
CG_order = 3
Ja_order = 4


def qammod(data, M):
    """Matlab-like square QAM modulation"""
    m = int(np.sqrt(M))
    re = 2 * (data % m) - m + 1
    im = 2 * (data // m) - m + 1
    symbols = re + 1j * im
    symbols = symbols / np.sqrt((2 / 3) * (M - 1))  # Norm avg power to 1
    return symbols


def qamdemod(symbols, M):
    """Matlab-like square QAM demodulation"""
    m = int(np.sqrt(M))
    scale = np.sqrt((2 / 3) * (M - 1))
    sy = symbols * scale

    re = np.real(sy)
    im = np.imag(sy)

    re_i = np.round((re + (m - 1)) / 2).astype(int)
    im_i = np.round((im + (m - 1)) / 2).astype(int)

    return im_i * m + re_i


def Neumann(A, b, iters):
    """Neumann Series approximation"""
    D = np.diag(np.diag(A))
    E = A - D
    D_inv = np.linalg.inv(D)

    Ainv = np.zeros_like(A, dtype=complex)

    # A^{-1} ≈ sum_{i=0}^k [ (-D^{-1}E)^i * D^{-1} ]
    DE = -D_inv @ E
    term = np.eye(A.shape[0], dtype=complex)

    for _ in range(iters + 1):
        Ainv += term @ D_inv
        term = term @ DE

    return Ainv @ b


def NI(A, b, iter, Nt):
    """Newton Iteration approximation"""
    D = np.diag(np.diag(A))
    Ainv = np.linalg.inv(D)  # initialize A0 inverse

    for i in range(iter):
        Ainv = Ainv @ (2 * np.eye(Nt, dtype=complex) - A @ Ainv)

    x_hat = Ainv @ b
    return x_hat


def Gauss(A, b, iters):
    """Gauss-Seidel iteration"""
    N = A.shape[0]
    x = np.zeros(N, dtype=complex)

    for _ in range(iters):
        for i in range(N):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x[i] = (b[i] - s1 - s2) / A[i, i]

    return x


def Jacobi(A, b, iter):
    """Jacobi iterative method"""
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    x_hat = D_inv @ b

    for i in range(iter):
        x_hat = D_inv @ (b + (D - A) @ x_hat)

    return x_hat


def CG(A, b, iter, Nt):
    """Conjugate Gradient method"""
    r = b.copy()
    p = r.copy()
    v = np.zeros(Nt, dtype=complex)

    for k in range(iter):
        e = A @ p
        alpha = np.linalg.norm(r) ** 2 / (p.conj().T @ e).real
        v = v + alpha * p

        new_r = r - alpha * e
        beta = np.linalg.norm(new_r) ** 2 / np.linalg.norm(r) ** 2
        p = new_r + beta * p
        r = new_r

    return v


# Channel generation
H = (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K)) * np.sqrt(0.5)

# Generate data and modulate
dataIn = np.random.randint(0, M, (Nt, K))
x = qammod(dataIn, M)

# Received signal without noise
y = np.zeros((Nr, K), dtype=complex)
for i in range(K):
    y[:, i] = H[:, :, i] @ x[:, i]

power_rx = np.mean(np.abs(x) ** 2)

# Initialize SER arrays
ser_NI = np.zeros(len(SNR))
ser_NS = np.zeros(len(SNR))
# ser_GS = np.zeros(len(SNR))
# ser_CG = np.zeros(len(SNR))
ser_Ja = np.zeros(len(SNR))


# Main simulation loop
for idx, snr in enumerate(SNR):
    log_noise = snr - 10 * np.log10(power_rx)

    sigma2 = 10 ** (-log_noise / 10)
    noise = (np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)) * np.sqrt(sigma2 / 2)
    y_noise = y + noise

    # Initialize detection arrays
    x_NI = np.zeros((Nt, K), dtype=complex)
    x_NS = np.zeros((Nt, K), dtype=complex)
    #    x_GS = np.zeros((Nt, K), dtype=complex)

    #    x_CG = np.zeros((Nt, K), dtype=complex)
    x_Ja = np.zeros((Nt, K), dtype=complex)

    # Process each symbol
    for j in range(K):
        Hj = H[:, :, j]
        A = Hj.conj().T @ Hj + sigma2 * np.eye(Nt)
        b = Hj.conj().T @ y_noise[:, j]

        # Apply different detection methods
        x_NI[:, j] = NI(A, b, NI_order, Nt)
        x_NS[:, j] = Neumann(A, b, NS_order)
        #        x_GS[:, j] = Gauss(A, b, GS_order)
        #        x_CG[:, j] = CG(A, b, CG_order, Nt)
        x_Ja[:, j] = Jacobi(A, b, Ja_order)

    # Demodulate
    dataOut_NI = qamdemod(x_NI, M)
    dataOut_NS = qamdemod(x_NS, M)
    #    dataOut_GS = qamdemod(x_GS, M)
    #    dataOut_CG = qamdemod(x_CG, M)
    dataOut_Ja = qamdemod(x_Ja, M)

    # Calculate SER
    ser_NI[idx] = np.sum(dataOut_NI != dataIn) / (Nt * K)

    ser_NS[idx] = np.sum(dataOut_NS != dataIn) / (Nt * K)
    #    ser_GS[idx] = np.sum(dataOut_GS != dataIn) / (Nt * K)

    #    ser_CG[idx] = np.sum(dataOut_CG != dataIn) / (Nt * K)
    ser_Ja[idx] = np.sum(dataOut_Ja != dataIn) / (Nt * K)

    print(f"SNR={snr} dB → SER_NI={ser_NI[idx]:.3e}, SER_NS={ser_NS[idx]:.3e}")

# Plot results
plt.figure(figsize=(10, 8))
plt.semilogy(SNR, ser_NS, "o-", label="Neumann Series")
plt.semilogy(SNR, ser_NI, "x-", label="Newton Iteration")
# plt.semilogy(SNR, ser_GS, "^-", label="Gauss Seidel")

plt.semilogy(SNR, ser_Ja, "d-", label="Jacobi method")
# plt.semilogy(SNR, ser_CG, "s-", label="Conjugate Gradient")
plt.semilogy(SNR, ser_adam, "*--", label="RMSProp")

plt.grid(True, which="both")
plt.xlabel("SNR [dB]")
plt.ylabel("SER")
plt.title(f"MIMO Detection (N×K={Nr}×{Nt}) - {M}-QAM {K}- Symbols")
plt.legend()
plt.ylim([1e-6, 1])
plt.tight_layout()
plt.show()
