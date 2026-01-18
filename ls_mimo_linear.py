import matplotlib.pyplot as plt
import numpy as np

# Parameters
Nr = 64  # Num of receiver's antennas (base station)

Nt = 8  # Num of transmitters (user equipments)

M = 64  # Use M-qam modulation
K = int(1e5)  # Num of symbols transmitted per user
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
ser_GS = np.zeros(len(SNR))
ser_CG = np.zeros(len(SNR))
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
    x_GS = np.zeros((Nt, K), dtype=complex)

    x_CG = np.zeros((Nt, K), dtype=complex)
    x_Ja = np.zeros((Nt, K), dtype=complex)

    # Process each symbol
    for j in range(K):
        Hj = H[:, :, j]
        A = Hj.conj().T @ Hj + sigma2 * np.eye(Nt)
        b = Hj.conj().T @ y_noise[:, j]

        # Apply different detection methods
        x_NI[:, j] = NI(A, b, NI_order, Nt)
        x_NS[:, j] = Neumann(A, b, NS_order)
        x_GS[:, j] = Gauss(A, b, GS_order)
        x_CG[:, j] = CG(A, b, CG_order, Nt)
        x_Ja[:, j] = Jacobi(A, b, Ja_order)

    # Demodulate
    dataOut_NI = qamdemod(x_NI, M)
    dataOut_NS = qamdemod(x_NS, M)
    dataOut_GS = qamdemod(x_GS, M)
    dataOut_CG = qamdemod(x_CG, M)
    dataOut_Ja = qamdemod(x_Ja, M)

    # Calculate SER
    ser_NI[idx] = np.sum(dataOut_NI != dataIn) / (Nt * K)

    ser_NS[idx] = np.sum(dataOut_NS != dataIn) / (Nt * K)
    ser_GS[idx] = np.sum(dataOut_GS != dataIn) / (Nt * K)

    ser_CG[idx] = np.sum(dataOut_CG != dataIn) / (Nt * K)
    ser_Ja[idx] = np.sum(dataOut_Ja != dataIn) / (Nt * K)

    print(
        f"SNR={snr} dB → SER_NI={ser_NI[idx]:.3e}, SER_NS={ser_NS[idx]:.3e}, SER_GS={
            ser_GS[idx]:.3e}"
    )

# Plot results
plt.figure(figsize=(10, 8))
plt.semilogy(SNR, ser_NS, "o-", label="Neumann Series")
plt.semilogy(SNR, ser_NI, "x-", label="Newton Iteration")
plt.semilogy(SNR, ser_GS, "^-", label="Gauss Seidel")

plt.semilogy(SNR, ser_Ja, "d-", label="Jacobi method")
plt.semilogy(SNR, ser_CG, "s-", label="Conjugate Gradient")

plt.grid(True, which="both")
plt.xlabel("SNR [dB]")
plt.ylabel("SER")
plt.title(f"MIMO Detection (N×K={Nr}×{Nt}) - {M}-QAM")
plt.legend()
plt.ylim([1e-6, 1])
plt.tight_layout()
plt.show()
