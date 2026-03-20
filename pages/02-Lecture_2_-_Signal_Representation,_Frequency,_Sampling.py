import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import src.helpers as helpers



st.set_page_config(page_title="Lecture 2 - Frequency Domain and Sampling", layout="wide")
st.title("Lecture 2: Frequency Domain and Sampling Theory")

DURATION = 5.0
FS_DEFAULT = 100


st.header("1. Time-domain vs frequency-domain")

st.markdown(r"""
A signal can be described in two useful ways:

- **Time-domain**: how the amplitude changes over time
- **Frequency-domain**: which sinusoidal components are present

A complicated waveform can often be understood as the sum of a few simple signals.
""")

col1, col2 = st.columns([0.28, 0.72])

with col1:
    fs1 = 25

    st.markdown("**Choose settings for signal 1:**")
    f_sine1 = st.slider("Frequency (Hz)", 0.5, 5.0, 3.0, 0.1, key="sine1")
    a_sine1 = st.slider("Amplitude", 0.1, 2.0, 3.0, 0.1, key="amp1")
    dc1 = st.slider("DC-offset", -2.0, 2.0, 0.0, 0.1, key="dc1")


    st.markdown("**Choose settings for signal 2:**")
    f_sine2 = st.slider("Frequency (Hz)", 0.5, 5.0, 4.5, 0.1, key="sine2")
    a_sine2 = st.slider("Amplitude", 0.1, 2.0, 4.5, 0.1, key="amp2")
    dc2 = st.slider("DC-offset", -2.0, 2.0, 0.0, 0.1, key="dc2")


with col2:
    duration = 5.0
    t = np.arange(0, duration, 1 / fs1)

    x1 = a_sine1 * np.sin(2 * np.pi * f_sine1 * t) + dc1
    x2 = a_sine2 * np.sin(2 * np.pi * f_sine2 * t) + dc2
    x_sum = x1 + x2

    freqs1, mag1, power1, _ = helpers.one_sided_spectrum(x1, fs1)
    freqs2, mag2, power2, _ = helpers.one_sided_spectrum(x2, fs1)
    freqs_sum, mag_sum, power_sum, _ = helpers.one_sided_spectrum(x_sum, fs1)

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    # top row: signals in time domain
    axs[0, 0].plot(t, x1, color="orange")
    axs[0, 0].set_title("Signal 1")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].set_xlim(0, duration)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t, x2, color="red")
    axs[0, 1].set_title("Signal 2")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].set_xlim(0, duration)
    axs[0, 1].grid(True, alpha=0.3)

    axs[0, 2].plot(t, x_sum, color="black")
    axs[0, 2].set_title("Superposition")
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Amplitude")
    axs[0, 2].set_xlim(0, duration)
    axs[0, 2].grid(True, alpha=0.3)

    # middle row: magnitude spectra
    axs[1, 0].stem(freqs1, mag1, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[1, 0].set_title("Magnitude spectrum 1")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Magnitude")
    axs[1, 0].set_xlim(0, fs1 / 2)
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].stem(freqs2, mag2, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[1, 1].set_title("Magnitude spectrum 2")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].set_xlim(0, fs1 / 2)
    axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].stem(freqs_sum, mag_sum, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[1, 2].set_title("Magnitude spectrum of superposition")
    axs[1, 2].set_xlabel("Frequency (Hz)")
    axs[1, 2].set_ylabel("Magnitude")
    axs[1, 2].set_xlim(0, fs1 / 2)
    axs[1, 2].grid(True, alpha=0.3)

    # bottom row: power spectra
    axs[2, 0].stem(freqs1, power1, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[2, 0].set_title("Power spectrum 1")
    axs[2, 0].set_xlabel("Frequency (Hz)")
    axs[2, 0].set_ylabel("Power")
    axs[2, 0].set_xlim(0, fs1 / 2)
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].stem(freqs2, power2, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[2, 1].set_title("Power spectrum 2")
    axs[2, 1].set_xlabel("Frequency (Hz)")
    axs[2, 1].set_ylabel("Power")
    axs[2, 1].set_xlim(0, fs1 / 2)
    axs[2, 1].grid(True, alpha=0.3)

    axs[2, 2].stem(freqs_sum, power_sum, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[2, 2].set_title("Power spectrum of superposition")
    axs[2, 2].set_xlabel("Frequency (Hz)")
    axs[2, 2].set_ylabel("Power")
    axs[2, 2].set_xlim(0, fs1 / 2)
    axs[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)




st.header("2. Period and frequency")

col1, col2 = st.columns([0.28, 0.72])

with col1:
    st.markdown(r"""
    For a periodic signal, the **period** $T$ is the time needed for one full cycle.

    The corresponding **frequency** is:

    $ f = \frac{1}{T} $,

    measured in Hertz.
    """)


    f2 = st.slider("Signal frequency (Hz)", 0.2, 5.0, 0.5, 0.01, key="sec2_f")
    T2 = 1 / f2

    st.badge("*Here we have:*", color="red")
    st.markdown(f"""
- frequency: **{f2:.2f} Hz**
- period: **{T2:.2f} s**
""")

with col2:
    t2 = np.linspace(0, DURATION, 2000)
    x2 = np.sin(2 * np.pi * f2 * t2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t2, x2, color="black")
    ax.axvline(1.0, linestyle="--", color="red")
    ax.axvline(1.0 + T2, linestyle="--", color="red")
    ax.annotate(
        "",
        xy=(1.0, 1.1),
        xytext=(1.0 + T2, 1.1),
        arrowprops=dict(arrowstyle="<->", color="red"),
    )
    ax.text(1.0 + T2 / 2, 1.15, f"T = {T2:.2f} s", ha="center", color="red")
    ax.set_title("A periodic sine wave")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, DURATION)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)



st.header("3. Sine matching and frequency detection")

st.markdown(r"""
A naive way to detect frequencies is to compare the signal with a **test sine** at many frequencies.
The correlation with the test sine will be strongest at frequencies that are actually present in the signal.
""")

col1, col2 = st.columns([0.28, 0.72])

with col1:
    fs3 = 50
    st.markdown(r"""
        Calculate correlation between signal and test sines at different frequencies:  
            $X[f] = corr(signal, \sin(2\pi f t))$
    
    """)

    st.markdown("**Choose settings of component 1:**")
    f_comp1 = st.slider("Frequency (Hz)", 0.5, 10.0, 3.0, 0.1, key="comp1_f")
    a_comp1 = st.slider("Amplitude", 0.1, 5.0, 2.0, 0.1, key="comp1_a")
    func_comp1 = st.selectbox("Function type", ["sin", "cos"], index=0, key="comp1_func")

    st.markdown("**Choose settings of component 2:**")
    f_comp2 = st.slider("Frequency (Hz)", 0.5, 10.0, 4.5, 0.1, key="comp2_f")
    a_comp2 = st.slider("Amplitude", 0.1, 5.0, 1.0, 0.1, key="comp2_a")
    func_comp2 = st.selectbox("Function type", ["sin", "cos"], index=0, key="comp2_func")



with col2:
    
    t = np.arange(0, 10, 1 / fs3)
    signal = a_comp1 * (np.sin(2 * np.pi * f_comp1 * t) if func_comp1 == "sin" else np.cos(2 * np.pi * f_comp1 * t))
    signal += a_comp2 * (np.sin(2 * np.pi * f_comp2 * t) if func_comp2 == "sin" else np.cos(2 * np.pi * f_comp2 * t))

    frequencies = np.arange(0.0, 10.0, 0.1)
    correlations = []

    for f_test in frequencies:
        test_sine = np.sin(2 * np.pi * f_test * t)
        corr = np.corrcoef(signal, test_sine)[0, 1]
        correlations.append(abs(corr))

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    axs[0].plot(t, signal, color="black")
    axs[0].set_title("Signal in time-domain")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, DURATION)
    axs[0].grid(True, alpha=0.3)    

    axs[1].stem(frequencies, correlations, linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[1].set_title("Correlation with test sines")
    axs[1].set_xlabel("Test frequency (Hz)")
    axs[1].set_ylabel("Correlation")
    axs[1].set_xlim(0, 10)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_ylim(-0.05, 1.05)

    plt.tight_layout() 
    st.pyplot(fig)

st.markdown(""" 
**Important**: this method is a) not very efficient, and b) not very robust as correlation
**critically** depends on the phase of the test sine. If you, for example, have a
cosine in your signal, correlation with test sines will be zero even at the corresponding
frequency (cosine and sine have a $\pi/2$-phase shift; they're orthogonal functions).
The DFT solves both issues elegantly.
""")


st.header("5. Reconstruction from Fourier coefficients")

col1, col2 = st.columns([0.38, 0.72])

with col1:
    st.markdown(r"""
    The inverse DFT **reconstructs** the signal from its Fourier coefficients.

    If we keep only the strongest coefficients, we get an **approximation**.
    The more coefficients we keep, the better the reconstruction. Perfect reconstruction
    is only achieved when we keep all coefficients.
    """)
    fs5 = 100

    t5 = np.arange(0, DURATION, 1 / fs5)
    x5 = (
        2.0 * np.sin(2 * np.pi * 3.0 * t5)
        + 1.0 * np.sin(2 * np.pi * 4.5 * t5)
        + 0.7 * np.cos(2 * np.pi * 8.0 * t5)
        + 0.5
    )

    keep_k =st.number_input(r"Keep $k$ components between 1 and 500 (signal length)", min_value=0, max_value=500, value=1, step=1)
    x5_recon, X5_sparse = helpers.reconstruct_from_top_k_fft(x5, keep_k)
    mse = np.mean((x5 - x5_recon) ** 2)
    st.latex(rf"\mathrm{{MSE}} = {mse:.16f}")

with col2:
    freqs5 = np.fft.fftfreq(len(X5_sparse), d=1 / fs5)
    order5 = np.argsort(freqs5)
    freqs5_sorted = freqs5[order5]
    X5_sparse_sorted = X5_sparse[order5]

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    axs[0].plot(t5, x5, color="black", label="Original")
    axs[0].plot(t5, x5_recon, linestyle="--", color="tab:red", label="Reconstruction")
    axs[0].set_title("Time-domain reconstruction")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, DURATION)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].stem(freqs5_sorted, np.abs(X5_sparse_sorted), linefmt="k-", markerfmt="k.", basefmt=" ")
    axs[1].set_title("Kept Fourier coefficients")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].set_xlim(-fs5 / 2, fs5 / 2)
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
        


st.header("6. Aliasing and the Nyquist criterion")

st.markdown(r"""
Aliasing occurs when the sampling rate is too low for the highest frequency in the signal.

To avoid aliasing, we need

$ f_s \ge 2 f_{\max} $

The value $f_s/2$ is the **Nyquist frequency**.
""")

col1, col2 = st.columns([0.28, 0.72])

with col1:
    fs7 = st.selectbox("Choose sampling rate", [100, 80, 50, 40], index=0, key="sec7_fs")
    nyq7 = fs7 / 2

    original_freqs = [3, 21, 27, 33]

    st.badge("*Original components:*", color="red")
    st.markdown(r"""The signal is a superposition of sines:  
$x(t) = \sin(2\pi \cdot 3 \cdot t) + 0.5 \sin(2\pi \cdot 21 \cdot t) + 0.5 \sin(2\pi \cdot 27 \cdot t) + 0.5 \sin(2\pi \cdot 33 \cdot t)$
""")

    st.markdown(f"""
- selected sampling rate: **{fs7} Hz**
- Nyquist frequency: **{nyq7:.1f} Hz**
""")

    aliased_freqs = [helpers.alias_frequency(f, fs7) for f in original_freqs]
    if fs7 >= 66:
        st.badge("No aliasing, sampling rate is high enough.", color="orange")
    else:
        st.badge("**Aliasing occurs:** some frequencies fold back into the visible interval.", color="red")
        st.markdown("**Aliased locations:**")
        for f_orig, f_alias in zip(original_freqs, aliased_freqs):
            st.markdown(f"- {f_orig} Hz → {f_alias:.1f} Hz")

with col2:
    dense_fs = 2000
    t_dense = np.arange(0, DURATION, 1 / dense_fs)
    t_samp = np.arange(0, DURATION, 1 / fs7)

    x_dense = (
        np.sin(2 * np.pi * 3 * t_dense)
        + 0.5*np.sin(2 * np.pi * 21 * t_dense)
        + 0.5*np.sin(2 * np.pi * 27 * t_dense)
        + 0.5*np.sin(2 * np.pi * 33 * t_dense)
    ) * (1 + 0.5 * t_dense)

    x_samp = (
        np.sin(2 * np.pi * 3 * t_samp)
        + 0.5*np.sin(2 * np.pi * 21 * t_samp)
        + 0.5*np.sin(2 * np.pi * 27 * t_samp)
        + 0.5*np.sin(2 * np.pi * 33 * t_samp)
    ) * (1 + 0.5 * t_samp)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # time-domain sampling view
    axs[0].plot(t_dense, x_dense, color="black", linewidth=1.2, alpha=0.85, label="Dense reference")
    axs[0].vlines(t_samp, 0, x_samp, color="red", alpha=0.35, linewidth=0.8)
    axs[0].plot(t_samp, x_samp, "ro", markersize=2.5, alpha=0.7, label="Samples")
    axs[0].set_title("Sampling a multi-frequency signal")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, DURATION)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # actual dft spectrum of sampled signal (two-sided)
    X = np.fft.fft(x_samp)
    freqs = np.fft.fftfreq(len(x_samp), d=1/fs7)
    X_shifted = np.fft.fftshift(X)
    freqs_shifted = np.fft.fftshift(freqs)
    magnitude = np.abs(X_shifted)

    axs[1].plot(freqs_shifted, magnitude, color="black")
    axs[1].set_title("Two-Sided Magnitude Spectrum of Sampled Signal")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Signal Magnitude")
    axs[1].set_xlim(-fs7/2, fs7/2)
    axs[1].grid(True, alpha=0.3)

    # conceptual frequency spikes
    axs[2].axvline(-nyq7, linestyle="--", color="gray")
    axs[2].axvline(nyq7, linestyle="--", color="gray")

    for f in original_freqs:
        if f <= nyq7:
            axs[2].vlines([f, -f], 0, 1, color="black", linewidth=2)
        else:
            fa = abs(helpers.alias_frequency(f, fs7))
            axs[2].vlines([fa, -fa], 0, 1, color="red", linewidth=2)

    axs[2].set_title("Visible frequencies after sampling (conceptual)")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Spike height")
    axs[2].set_xlim(-fs7 / 2, fs7 / 2)
    axs[2].set_ylim(0, 1.2)
    axs[2].grid(True, alpha=0.3)

    

    plt.tight_layout()
    st.pyplot(fig)

