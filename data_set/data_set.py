import numpy as np


def create_mimo_data(Nr, Nt, M, K, number_type=""):
    angles = 2 * np.pi * np.arange(M) / M

    if number_type == "complex":
        PSK_Table = np.exp(1j * angles)
        H = np.sqrt(0.5) * (
            np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K)
        )
    else:
        PSK_Table = np.exp(angles)
        H = np.sqrt(0.5) * (np.random.randn(Nr, Nt, K))

    num_combinations = M**Nt
    z = np.zeros((Nt, num_combinations), dtype=int)
    z = np.array(np.unravel_index(np.arange(M**Nt), (M,) * Nt))
    combinML = PSK_Table[z]

    dataIn = np.random.randint(0, M, (Nt, K))
    dataMod = PSK_Table[dataIn]
    x = dataMod

    y = np.einsum("rnk,nk->rk", H, x)

    return H, y, x, dataIn
