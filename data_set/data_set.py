import numpy as np


def create_mimo_data(Nr, Nt, M, K, SNR, num_combinations):
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

    snr_db = np.random.uniform(0, 20)
    snr_feature = np.full((K, 1), snr_db / 20.0)
    X = np.hstack(
        [z.real, z.imag, H.real.reshape(K, -1), H.imag.reshape(K, -1), snr_feature]
    )

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_mean = (X - mu) / sigma
    X_train_mean = X.mean(axis=0)
    X_train_std = X.std(axis=0) + 1e-8
    X = X_mean

    y_labels = np.ravel_multi_index(dataIn, (M,) * Nt)

    return H, X, X_train_mean, X_train_std, x, y, y_labels, dataIn
