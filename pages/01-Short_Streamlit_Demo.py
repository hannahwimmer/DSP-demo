import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="DSP Demo", layout="wide")
st.title("Digital Signal Processing Demo")
st.image("docs/images/overview.png")

st.markdown("""
Here's a quick Streamlit demo to showcase its use for creating interactive demos. As
an example, we'll briefly take a look at two fundamental DSP concepts:

1. **Sampling**  
   We'll explore how the sampling rate affects the discrete-time representation.

2. **Superposition**  
   We'll build a more complex signal by adding several pure tones together.
""")



st.header("Introduction")

cont = st.container(border=True)
with cont:
    st.badge("Main concepts:", color="blue")
    st.markdown("""
    - A **continuous-time signal** can be represented by a smooth curve
    - A **sampled signal** consists of discrete values taken at regular time intervals
    - A complex periodic signal can often be built from the **sum of simple sinusoids**
    """)

    st.badge("Things you should notice:", color="blue")
    st.markdown("""
    - Increasing the sampling rate gives a more faithful representation
    - If the sampling rate is too low, the sampled points can be misleading
    - Different amplitudes, frequencies, and phases change the final composed signal
    """)

st.header("1. Sampling a Pure Sine Wave")

st.markdown("""
With streamlit, you can very easily create interactive controls to manipulate parameters
and see the results in real time - that's what we'll use to quickly explore the effects
of tinking with aspects of the topics we'll talk about. In this first part, we'll look
at how sampling a pure sine wave works.
""")

col1, col2 = st.columns([0.32, 0.68])

with col1:
    st.subheader("Controls")

    sine_freq = st.slider("Sine frequency (Hz)", 1.0, 20.0, 5.0, 0.5)
    sine_amp = st.slider("Amplitude", 0.2, 3.0, 1.0, 0.1)
    fs = st.slider("Sampling rate fs (Hz)", 2, 100, 12, 1)

    nyquist = fs / 2
    aliasing = sine_freq > nyquist

with col2:

    def sine_wave(t, amplitude=1.0, frequency=1.0, phase_deg=0.0):
        phase_rad = np.deg2rad(phase_deg)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase_rad)

    def make_sample_times(duration, fs):
        n_samples = max(2, int(fs * duration))
        return np.linspace(0, duration, n_samples, endpoint=False)

    t_cont = np.linspace(0, 1, 2000)
    y_cont = sine_wave(t_cont, amplitude=sine_amp, frequency=sine_freq, phase_deg=0)

    t_samp = make_sample_times(1, fs)
    y_samp = sine_wave(t_samp, amplitude=sine_amp, frequency=sine_freq, phase_deg=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t_cont, y_cont, color="black", label="Continuous signal")
    ax.stem(t_samp, y_samp, linefmt="r-", markerfmt="ro", basefmt=" ", label="Samples")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sampling of a Pure Sine Wave")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

st.markdown("### Quick interpretation")
st.write(f"- Sampling interval: **{1/fs:.4f} s**")
st.write(f"- Nyquist frequency: **{nyquist:.2f} Hz**")

if aliasing:
    st.error(
        f"Aliasing warning: the signal frequency ({sine_freq:.2f} Hz) "
        f"is above the Nyquist frequency ({nyquist:.2f} Hz)."
    )
else:
    st.success(
        f"No aliasing expected: {sine_freq:.2f} Hz is below the Nyquist frequency."
    )

st.header("2. Superposition of Sinusoids")

st.markdown("""
A composed signal can be created by **adding multiple pure sinusoids** (we use the
reverse of this fact later on when identifying frequency content in signals).
""")

left, right = st.columns([0.38, 0.62])

with left:
    st.subheader("Composition setup")

    mode = st.selectbox(
        "Choose a setup",
        ["Harmonics", "Beating"]
    )

    show_components = st.checkbox("Show individual components", value=True)

if mode == "Harmonics":
    amps = [0.5,0.5,0.5]
    freqs = [
        st.slider("Frequency component 1 (Hz)", 0, 10, 6, 1),
        st.slider("Frequency component 2 (Hz)", 0, 10, 3, 1),
        st.slider("Frequency component 3 (Hz)", 0, 10, 9, 1),
    ]

elif mode == "Beating":
    st.info("Two nearby frequencies create a beating effect.")
    f_center = st.slider("Center frequency (Hz)", 2.0, 20.0, 8.0, 0.5)
    delta = st.slider("Frequency separation (Hz)", 0.1, 3.0, 0.6, 0.1)

    freqs = [f_center - delta / 2, f_center + delta / 2, 0.0]
    amps = [
        st.slider("Amplitude component 1", 0.0, 2.0, 1.0, 0.1, key="b_a1"),
        st.slider("Amplitude component 2", 0.0, 2.0, 1.0, 0.1, key="b_a2"),
        0.0,
    ]

else:
    n_components = st.slider("Number of sinusoids", 1, 4, 3)

    freqs = []
    amps = []
    phases = [0,0,0]

    for i in range(n_components):
        st.markdown(f"**Component {i+1}**")
        amps.append(st.slider(f"Amplitude {i+1}", 0.0, 2.0, 1.0 if i == 0 else 0.5, 0.1, key=f"c_a_{i}"))
        freqs.append(st.slider(f"Frequency {i+1} (Hz)", 0.5, 20.0, float(2 * i + 2), 0.5, key=f"c_f_{i}"))

with right:
    t = np.linspace(0, 1, 2500)
    components = []
    composed = np.zeros_like(t)
    phases = [0,0,0]

    for a, f, p in zip(amps, freqs, phases):
        if a != 0 and f != 0:
            y = sine_wave(t, amplitude=a, frequency=f, phase_deg=p)
        else:
            y = np.zeros_like(t)
        components.append(y)
        composed += y

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    if show_components:
        for i, y in enumerate(components):
            if np.any(np.abs(y) > 1e-12):
                ax2.plot(t, y, linewidth=1.4, alpha=0.8, label=f"Component {i+1}")

    ax2.plot(t, composed, color="black", linewidth=2.5, label="Sum / Composed signal")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Superposition of Sinusoids")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)


st.header("Discussion")

st.markdown("""
You got a quick overview of how to use Streamlit to create interactive demos. As for the
showcased topics, we quickly summarize our findings in a Discussion section. How you
do it is up to you - check out the Streamlit documentation for more ideas.
""")

cont = st.container(border=True)
with cont:
    st.badge("Key takeaways:", color="blue")
    st.markdown("""
    - A higher sampling rate generally captures a signal more accurately.
    - The Nyquist frequency is half the sampling rate.
    - If the signal frequency exceeds the Nyquist frequency, **aliasing** can occur.
    - Adding sinusoids with different amplitudes, frequencies, and phases creates richer signals.
    - Harmonics and beating are especially nice visual examples for teaching DSP intuition.
    """)