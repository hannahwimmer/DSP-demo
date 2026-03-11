import numpy as np
import matplotlib.pyplot as plt


def stem_plot(ax, n, x, title="", xlabel="n", ylabel="Amplitude", color=None, ylim=None):
    markerline, stemlines, baseline = ax.stem(n, x, basefmt=" ")
    if color is not None:
        plt.setp(markerline, color=color)
        plt.setp(stemlines, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

def max_abs(*arrays):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.size > 0:
            vals.append(np.max(np.abs(arr)))
    m = max(vals) if vals else 1.0
    return max(m, 1e-6)

def delta_sequence(n, n0=0):
    return (n == n0).astype(float)

def shift_signal_by_samples(x, k):
    # y[n] = x[n-k] : delay by k samples
    y = np.zeros_like(x)
    if k > 0:
        y[k:] = x[:-k]
    elif k < 0:
        y[:k] = x[-k:]
    else:
        y = x.copy()
    return y

def flip_signal(x):
    # visual reversal of sequence values
    return x[::-1]

def index_scale_signal(x, factor=2):
    # y[n] = x[factor*n]
    y = np.zeros_like(x)
    N = len(x)
    for i in range(N):
        src = factor * i
        if 0 <= src < N:
            y[i] = x[src]
    return y

def moving_average_full(x, m=3):
    h = np.ones(m) / m
    y = np.convolve(x, h, mode="full")
    return y, h

def forward_difference_full(x):
    h = np.array([-1, 1], dtype=float)
    y = np.convolve(x, h, mode="full")
    return y, h

def apply_system_by_name(x, system_name, shift_k=3, scale_factor=2):
    if system_name == "Shift":
        return shift_signal_by_samples(x, shift_k)
    elif system_name == "Flip":
        return flip_signal(x)
    elif system_name == "Scale":
        return index_scale_signal(x, scale_factor)
    elif system_name == "Moving average":
        y, _ = moving_average_full(x, 3)
        return y
    elif system_name == "Forward difference":
        y, _ = forward_difference_full(x)
        return y
    else:
        return x.copy()

def test_superposition(system_func, x1, x2, alpha=1.0, beta=1.0):
    left = system_func(alpha * x1 + beta * x2)
    right = alpha * system_func(x1) + beta * system_func(x2)
    ok = np.allclose(left, right, atol=1e-9, rtol=1e-9)
    return left, right, ok

def build_common_axis(x, h):
    N = len(x) + len(h) - 1
    return np.arange(N)

def shift_kernel_for_convolution(h, out_idx, N):
    # Build h[n0-k] over k=0,...,N-1
    h_shifted = np.zeros(N)
    for k in range(N):
        h_idx = out_idx - k
        if 0 <= h_idx < len(h):
            h_shifted[k] = h[h_idx]
    return h_shifted


