import numpy as np

# index
dataIn = np.random.randint(0, 2, (2, 5))

print(dataIn)
print()
dataIn = np.random.randint(0, 2, (2, 100000))
print("\nDataIn\n", dataIn[:5], len(dataIn))


print("\n-------------creating the mimo data-----------------\n")


Nr = 2
Nt = 2
M = 2
K = int(1e3)
SNR = np.arange(0, 22, 2)
num_combinations = M**Nt

angles = 2 * np.pi * np.arange(M) / M
PSK_Table = np.exp(1j * angles)

z = np.zeros((Nt, num_combinations), dtype=int)
z = np.array(np.unravel_index(np.arange(M**Nt), (M,) * Nt))
combinML = PSK_Table[z]

H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))

dataIn = np.random.randint(0, M, (Nt, K))
dataMod = PSK_Table[dataIn]
x = dataMod

y = np.einsum("rnk,nk->rk", H, x)

# H_samples = np.transpose(H, (2, 0, 1))
# X = H_samples.reshape(K, Nr * Nt)
# y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)

y_samples = y.T
print(
    "\n----------------y_samples------------\n",
    y_samples[:5],
    y_samples.shape,
)

H_samples = np.transpose(H, (2, 0, 1)).reshape(K, Nr * Nt)
print("\n---------H_samples----------\n", H_samples[:5], H_samples.shape)

X = np.hstack([y_samples, H_samples])
print("\n------------X---------------\n", X[:5], X.shape)

y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)
print("\n------------y_labels----------------\n", y_labels[:5], y_labels.shape)

# @ is matrix multiplication (@ is batch aware), for vetcors it's simplye the dot product
# same as np.matmul, its matrix multiplication
# * is element wise multiplication
z = H.conj().transpose(0, 2, 1) @ y
print("\n------------z-----------------\n", z[:5], z.shape)

h_real_reshape = H.real.reshape(K, -1)
print(
    "\n----------H real, complex reshape-----------\n",
    h_real_reshape[:5],
    h_real_reshape.shape,
    H.shape,
)


h_imag_reshape = H.imag.reshape(K, -1)
print(
    "\n----------H real, complex reshape-----------\n",
    h_imag_reshape[:5],
    h_imag_reshape.shape,
    H.shape,
)

print("\n-----Shapes------\n")
print(
    y.T.shape,
    y.T.real.shape,
    y.T.imag.shape,
    H.shape,
    H.real.reshape(K, -1).shape,
    H.real.reshape(K, -1).shape,
    z.shape,
)

# When calling hstack, all arrays must have the same exact number of rows
X = np.hstack([y.T.real, y.T.imag, H.real.reshape(K, -1), H.imag.reshape(K, -1)])
print("\n-------The result hstack----------\n", X[:5], X.shape)

# Batch, per sample conjugate MF
H_batched = np.transpose(H, (2, 0, 1))  # (K, Nr, Nt)
y_batched = y.T  # (K, Nr)
Hh = H_batched.conj().transpose(0, 2, 1)  # (K, Nt, Nr)
y_col = y_batched[..., None]  # (K, Nr, 1)
z = Hh @ y_col  # (K, Nt, 1)
z = z.squeeze(-1)  # removes last dimension
print("\n-------------z batch-----------\n", z[:5], z.shape)

X = np.hstack([z.real, z.imag, H.real.reshape(K, -1), H.imag.reshape(K, -1)])
print("\n-------The result hstack z----------\n", X[:5], X.shape)

# now normalize X
mu = X.mean(axis=0)
sigma = X.std(axis=0) + 1e-8
X = (X - mu) / sigma

print("\n-------The result hstack z, X normed----------\n", X[:5], X.shape)

y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)
z_labels = np.ravel_multi_index(dataIn, (M,) * Nt)
print(
    "\n-------------Label shape checking-----------\n",
    y_labels[:10],
    y_labels.shape,
    z_labels[:10],
    z_labels.shape,
)

"""
z values are larger magnitude → matched filter concentrating energy

H values are roughly unit variance → channel normalization is intact

No NaNs, no infinities, no constant columns

"""

# z = H.conj().transpose(0, 2, 1) @ y[..., None]
# the none at the end adds an extra dimension of size 1 so that multiplication gets right
# (A @ B): (..., m, n) @ (..., n, p) → (..., m, p)
# H.shape        == (K, Nr, Nt)
# Hᴴ.shape       == (K, Nt, Nr)
# y.shape        == (K, Nr)       not a matrixk
# y[..., None] means keep all existing dimensions as is

# print("\n", H.shape, " ", x.shape)
# print("\n", H_samples[:5], H_samples.shape)
# print("\n", y_labels[:5], y_labels.shape)
# print("\n", X[:5], X.shape)
# print("\ndataIn\n", dataIn[:5], dataIn.shape)
