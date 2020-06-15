"""
Module: LibFMP.C2.fourier
Author: Frank Zalkow, Meinard Müller
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP).
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def generate_matrix_dft(N, K):
    """Generates a DFT (discete Fourier transfrom) matrix

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        N: Number of samples
        K: Number of frequency bins

    Returns:
        dft: The DFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(-2j * np.pi * k * n / N)
    return dft


@jit(nopython=True)
def generate_matrix_dft_inv(N, K):
    """Generates an IDFT (inverse discete Fourier transfrom) matrix

    Notebook: C2/C2_STFT-Inverse.ipynb

    Args:
        N: Number of samples
        K: Number of frequency bins

    Returns:
        dft: The IDFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(2j * np.pi * k * n / N) / N
    return dft


@jit(nopython=True)
def dft(x):
    """Compute the discete Fourier transfrom (DFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        x: Signal to be transformed

    Returns:
        X: Fourier transform of `x`
    """
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft(N, N)
    return np.dot(dft_mat, x)


@jit(nopython=True)
def idft(x):
    """Compute the inverse discete Fourier transfrom (IDFT)

    Notebook: C2/C2_STFT-Inverse.ipynb

    Args:
        x: Signal to be transformed

    Returns:
        X: Fourier transform of `x`
    """
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft_inv(N, N)
    return np.dot(dft_mat, x)


@jit(nopython=True)
def twiddle(N):
    """Generate the twiddle factors used in the computation of the fast Fourier transform (FFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        N: Number of samples

    Returns:
        sigma: The twiddle factors
    """
    k = np.arange(N // 2)
    sigma = np.exp(-2j * np.pi * k / N)
    return sigma


@jit(nopython=True)
def twiddle_inv(N):
    """Generate the twiddle factors used in the computation of the Inverse fast Fourier transform (IFFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        N: Number of samples

    Returns:
        sigma: The twiddle factors
    """
    n = np.arange(N // 2)
    sigma = np.exp(2j * np.pi * n / N)
    return sigma


@jit(nopython=True)
def fft(x):
    """Compute the fast Fourier transform (FFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        x: Signal to be transformed

    Returns:
        X: Fourier transform of `x`
    """
    x = x.astype(np.complex128)
    N = len(x)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    X = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return x
    else:
        this_range = np.arange(N)
        A = fft(x[this_range % 2 == 0])
        B = fft(x[this_range % 2 == 1])
        C = twiddle(N) * B
        X[:N//2] = A + C
        X[N//2:] = A - C
        return X


@jit(nopython=True)
def ifft_noscale(X):
    """Compute the inverse fast Fourier transform (IFFT) without the final scaling factor of 1/N

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        X: Fourier transform of `x`

    Returns:
        x: Inverse Fourier transform of `x`
    """
    X = X.astype(np.complex128)
    N = len(X)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    x = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return X
    else:
        this_range = np.arange(N)
        A = ifft_noscale(X[this_range % 2 == 0])
        B = ifft_noscale(X[this_range % 2 == 1])
        C = twiddle_inv(N) * B
        x[:N//2] = A + C
        x[N//2:] = A - C
        return x


@jit(nopython=True)
def ifft(X):
    """Compute the inverse fast Fourier transform (IFFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        X: Fourier transform of `x`

    Returns:
        x: Inverse Fourier transform of `x
    """
    return ifft_noscale(X) / len(X)


@jit(nopython=True)
def stft(x, w, H=512, zero_padding=0, only_positive_frequencies=False):
    """Compute the discrete short-time Fourier transform (STFT)

    Notebook: C2/C2_STFT-Basic.ipynb

    Args:
        x: Signal to be transformed
        w: Window function
        H: Hopsize
        zero_padding: Number of zeros to be padded after windowing and before the Fourier transform of a frame
        only_positive_frequencies: Only return positive frequency part of spectrum (non-invertible)

    Returns:
        X: The discrete short-time Fourier transform
    """

    N = len(w)
    x = np.concatenate((np.zeros(N // 2), x, np.zeros(N // 2)))

    L = len(x)
    M = int(np.floor((L - N) / H))

    X = np.zeros((N + zero_padding, M + 1), dtype=np.complex128)
    zero_padding_vector = np.zeros((zero_padding, ), dtype=x.dtype)

    for m in range(M + 1):
        x_win = x[m * H:m * H + N] * w
        if zero_padding > 0:
            x_win = np.concatenate((x_win, zero_padding_vector))
        X_win = fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = (N + zero_padding + 1) // 2
        X = X[:K, :]
    return X


@jit(nopython=True)
def istft(X, w, H, L, zero_padding=0):
    """Compute the inverse discrete short-time Fourier transform (ISTFT)

    Notebook: C2/C2_STFT-Inverse.ipynb

    Args:
        X: The discrete short-time Fourier transform
        w: Window function
        H: Hopsize
        L: Length of time signal
        zero_padding: Number of zeros to be padded after windowing and before the Fourier transform of a frame

    Returns:
        x_rec: Reconstructed time signal
    """
    N = len(w)
    L = L + N
    M = X.shape[1]
    w_sum = np.zeros(L)
    x_win_sum = np.zeros(L)
    w_sum = np.zeros(L)
    for m in range(M):
        start_idx, end_idx = m * H, m * H + N + zero_padding
        if start_idx > L:
            break

        x_win = ifft(X[:, m])
        if end_idx > L:
            end_idx = L
            x_win = x_win[:end_idx-start_idx]
            cur_w = w[:end_idx-start_idx]
        else:
            cur_w = w

        # Avoid imaginary values (due to floating point arithmetic)
        x_win_real = np.real(x_win)
        x_win_sum[start_idx:end_idx] = x_win_sum[start_idx:end_idx] + x_win_real
        w_shifted = np.zeros(L)
        w_shifted[start_idx:start_idx + len(cur_w)] = cur_w
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum / w_sum
    x_rec = x_rec[N // 2:-N // 2]
    return x_rec
