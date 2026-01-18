import matplotlib.pyplot as plt
import numpy as np

# print(matplotlib.get_backend())
# matplotlib.use("webagg")

Nr = 64  # Recieving antennas
Nt = 8  # Transmitting antennas
M = 2  # M-PSK modulation
K = int(1e5)  # number of symbols transmitted per reciever
SNR = np.arange(0, 22, 2)
bit_number = Nt * K * np.log2(M)

# Generate a table of M-PSK constellation
angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(1j * angles)

# Init array to hold all combinations
num_combinations = M**Nt
z = np.zeros((Nt, num_combinations), dtype=int)

# Generate an array to hold PSK table indicies
z = np.array(np.unravel_index(np.arange(M**Nt), (M,) * Nt))

# ML detector Array
combinML = PSK_Table[z]

# Channel Matrix (Nr * Nt * K)
H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))

# Random Transmitted sybmols (Nt * K)
dataIn = np.random.randint(0, M, (Nt, K))

# PSK Modulation
dataMod = PSK_Table[dataIn]

# Transmitted symbols (Nt * K)
x = dataMod

# Output y = Hx
# This computes: y[:, i] = H[:, :, i] * x[:, i]
y = np.einsum("rnk,nk->rk", H, x)

# Signal energy
power_rx = np.mean(np.abs(x) ** 2)

# Init SER arrays
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

    # Demodulate

    # x_MMSE[:, :, None]            # shape → (Nt, K, 1)
    # PSK_Table[None, None, :]      # shape → (1, 1, M)
    # creates a 3D array making shape (Nt, K, M)
    # calculating constellation s_hat=argmin_m abs((x - c_m)**2)

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

        # Compute distance || y - H * candidate ||^2 for all candidates
        # combin_ML: Nt × (M^Nt)
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
plt.semilogy(SNR, ser_MMSE, "o--", label="MMSE")
plt.semilogy(SNR, ser_ZF, "+--", label="ZF")
plt.semilogy(SNR, ser_ML, "--.", label="ML")

plt.grid(True)
plt.xlabel("SNR [dB]")
plt.ylabel("SER")
plt.title("Small Mimo Detection, Nr(64)*Nt(8), BPSK, 100k symbols")
plt.legend()

plt.show()
