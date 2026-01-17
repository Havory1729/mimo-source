import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("webagg")

Nr = 64  # Num of receiver's antennas (base station)
Nt = 8  # Num of transmitters (user equipments)
M = 64  # Use M-qam modulation
K = int(1e5)  # Num of symbols transmitted per user
SNR = np.arange(5, 15)
bit_number = int(Nt * K * np.log2(M))
NS_order = 3
GS_order = 3

# TODO: Expalin qammod, qamdemod, Neuman, Gauss-Seidal


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
    D = np.diag(np.diag(A))
    E = A - D
    D_inv = np.linalg.inv(D)

    Ainv = np.zeros_like(A)

    # A^{-1} ≈ sum_{i=0}^k [ (-D^{-1}E)^i * D^{-1} ]
    DE = -D_inv @ E
    term = np.eye(A.shape[0])

    for _ in range(iters + 1):
        Ainv += term @ D_inv
        term = term @ DE

    return Ainv @ b


def Gauss(A, b, iters):
    N = A.shape[0]
    x = np.zeros(N, dtype=complex)

    for _ in range(iters):
        for i in range(N):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x[i] = (b[i] - s1 - s2) / A[i, i]

    return x


# Channel generation
# H = np.zeros((Nt * Nr * K), dtype=complex)
H = (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K)) * np.sqrt(0.5)

dataIn = np.random.randint(0, M, (Nt, K))
x = qammod(dataIn, M)

y = np.zeros((Nr, K), dtype=complex)
# for i in range(K):
#     y[:, i] = H[:, :, i] @ x[:, i]

y = np.einsum("rnk,nk->rk", H, x)

power_rx = np.mean(np.abs(x) ** 2)

# Init SER
ser_NS = np.zeros(len(SNR))
ser_GS = np.zeros(len(SNR))

# Main sim loop
for idx, snr in enumerate(SNR):
    log_noise = snr - 10 * np.log10(power_rx)
    sigma2 = 10 ** (-log_noise / 10)

    noise = (np.random.randn(Nr, K) + 1j * np.random.randn(Nr, K)) * np.sqrt(sigma2 / 2)

    y_noise = y + noise

    x_NS = np.zeros((Nt, K), dtype=complex)

    x_GS = np.zeros((Nt, K), dtype=complex)

    for j in range(K):
        Hj = H[:, :, j]
        A = Hj.conj().T @ Hj + sigma2 * np.eye(Nt)
        b = Hj.conj().T @ y_noise[:, j]

        x_NS[:, j] = Neumann(A, b, NS_order)
        x_GS[:, j] = Gauss(A, b, GS_order)

    # Detection
    dataOut_NS = qamdemod(x_NS, M)
    dataOut_GS = qamdemod(x_GS, M)

    # SER computation
    ser_NS[idx] = np.sum(dataOut_NS != dataIn) / (Nt * K)
    ser_GS[idx] = np.sum(dataOut_GS != dataIn) / (Nt * K)

    # print(f"SNR={snr} dB → SER_NS={ser_NS[idx]:.3e}, SER_GS={ser_GS[idx]:.3e}")


plt.figure(figsize=(8, 6))
plt.semilogy(SNR, ser_NS, "o-", label="Neumann Series")
plt.semilogy(SNR, ser_GS, "^-", label="Gauss-Seidel")

plt.grid(True, which="both")
plt.xlabel("SNR [dB]")
plt.ylabel("SER")
plt.title("MIMO Detection (64 × 8), 64-QAM")
plt.legend()
plt.show()
