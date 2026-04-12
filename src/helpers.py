import numpy as np
import matplotlib.pyplot as plt


# for session 1
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


# for session 2
# for session 2
def safe_corr(a, b):
    """
    Normalized zero-mean correlation / inner product.
    Good for showing why sine-only matching can fail for cosine signals.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a - np.mean(a)
    b = b - np.mean(b)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    return np.dot(a, b) / (na * nb)


def build_signal(t, components, dc=0.0, noise_std=0.0, seed=0):
    """
    Build a signal from components.

    components: list of tuples
        (amplitude, frequency_hz, basis)
    where basis is "sin" or "cos"
    """
    x = np.zeros_like(t, dtype=float) + dc

    for amp, freq, basis in components:
        if basis == "sin":
            x += amp * np.sin(2 * np.pi * freq * t)
        else:
            x += amp * np.cos(2 * np.pi * freq * t)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        x += rng.normal(0, noise_std, size=len(t))

    return x


def one_sided_spectrum(signal, fs):
    """
    One-sided spectrum for real-valued signals.
    Returns:
        freqs, magnitude, power, X
    """
    signal = np.asarray(signal, dtype=float)
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
    magnitude = np.abs(X)
    power = magnitude ** 2
    return freqs, magnitude, power, X


def full_spectrum(signal, fs):
    """
    Full FFT spectrum including negative frequencies.
    Returns sorted frequencies and FFT values.
    """
    signal = np.asarray(signal, dtype=float)
    X = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)
    order = np.argsort(freqs)
    return freqs[order], X[order]


def alias_frequency(f, fs):
    """
    Fold frequency into the Nyquist interval [-fs/2, fs/2].
    """
    return ((f + fs / 2) % fs) - fs / 2


def parseval_energy(signal):
    """
    Returns:
        energy_time, energy_freq
    """
    signal = np.asarray(signal, dtype=float)
    X = np.fft.fft(signal)
    energy_time = np.sum(np.abs(signal) ** 2)
    energy_freq = np.sum(np.abs(X) ** 2) / len(X)
    return energy_time, energy_freq


def reconstruct_from_top_k_fft(signal, k):
    """
    Keep only the k strongest FFT bins and reconstruct.
    Returns:
        x_recon, X_sparse
    """
    signal = np.asarray(signal, dtype=float)
    X = np.fft.fft(signal)

    idx_sorted = np.argsort(np.abs(X))[::-1]
    keep_idx = idx_sorted[:k]

    X_sparse = np.zeros_like(X, dtype=complex)
    X_sparse[keep_idx] = X[keep_idx]

    x_recon = np.fft.ifft(X_sparse).real
    return x_recon, X_sparse


def make_short_window(window_type: str, N: int) -> np.ndarray:
    if window_type == "Rectangular":
        return np.ones(N)
    elif window_type == "Hann":
        return np.hanning(N)
    elif window_type == "Hamming":
        return np.hamming(N)
    elif window_type == "Blackman":
        return np.blackman(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    

def place_window_on_signal_grid(window: np.ndarray, total_len: int, start_idx: int) -> np.ndarray:
    y = np.zeros(total_len)
    end_idx = min(start_idx + len(window), total_len)
    valid_len = max(0, end_idx - start_idx)
    if valid_len > 0:
        y[start_idx:end_idx] = window[:valid_len]
    return y


def make_analysis_window(window_type: str, N: int) -> np.ndarray:
    if window_type == "Rectangular":
        return np.ones(N)
    elif window_type == "Hann":
        return np.hanning(N)
    elif window_type == "Hamming":
        return np.hamming(N)
    elif window_type == "Blackman":
        return np.blackman(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def place_window(window: np.ndarray, total_len: int, start_idx: int) -> np.ndarray:
    y = np.zeros(total_len)
    end_idx = min(start_idx + len(window), total_len)
    valid_len = max(0, end_idx - start_idx)
    if valid_len > 0:
        y[start_idx:end_idx] = window[:valid_len]
    return y


def generate_scenario_signal(scenario: str, t: np.ndarray, fs: float, rng: np.random.Generator,
                             noise_std: float, delta_f: float) -> tuple[np.ndarray, str]:
    if scenario == "Noisy single tone":
        x = np.sin(2 * np.pi * 20 * t) + noise_std * rng.normal(size=len(t))
        msg = "A single tone in broadband noise."
    elif scenario == "Strong distant interferer":
        x = 0.2 * np.sin(2 * np.pi * 20 * t) + 1.0 * np.sin(2 * np.pi * 60 * t)
        msg = "Weak tone at 20 Hz, strong interferer at 60 Hz."
    elif scenario == "Strong nearby interferer":
        x = 0.25 * np.sin(2 * np.pi * 20 * t) + 1.0 * np.sin(2 * np.pi * (20 + delta_f) * t)
        msg = f"Weak tone at 20 Hz, strong nearby interferer at {20 + delta_f:.1f} Hz."
    elif scenario == "Two close equal tones":
        x = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * (20 + delta_f) * t)
        msg = f"Two close tones at 20 Hz and {20 + delta_f:.1f} Hz."
    elif scenario == "Two close unequal tones":
        x = 1.0 * np.sin(2 * np.pi * 20 * t) + 0.45 * np.sin(2 * np.pi * (20 + delta_f) * t)
        msg = f"Unequal close tones at 20 Hz and {20 + delta_f:.1f} Hz."
    elif scenario == "Broadband / flat spectrum":
        x = rng.normal(size=len(t))
        msg = "Broadband random signal."
    else:
        x = np.sin(2 * np.pi * 20 * t)
        msg = "Fallback signal."
    return x, msg