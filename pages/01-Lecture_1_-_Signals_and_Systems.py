import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import src.helpers as helpers


st.set_page_config(page_title="Lecture 1 - Signals and Systems", layout="wide")
st.title("Lecture 1: Signals and Systems <3")

st.header("1. What is a signal?")

st.markdown("""
A **signal** is a function that carries information. In this course, we often use
discrete-time signals, written as $x[n]$.
""")

st.badge("**This means:**", color="red")
st.markdown("""
    - the signal is defined only at integer indices $n$  
    *we don't have any information about what happens at values in-between*
    - the signal values at the available indices comprise the sequence $\{x[n]\}$
""")

col1, col2 = st.columns([0.3, 0.7])

with col1:
    fs = st.slider("Sampling rate (Hz)", 1, 40, 8, 1)

    duration = 5.0
    f0 = 0.5
    t_cont = np.linspace(0, duration, 2000)
    x_cont = np.sin(2 * np.pi * f0 * t_cont)

    t_samp = np.arange(0, duration + 1e-12, 1 / fs)
    x_samp = np.sin(2 * np.pi * f0 * t_samp)

    st.badge("*Here we have:*", color="red")
    st.markdown(f"""
        - signal frequency: **{f0} Hz**
        - duration: **{duration} seconds**
        - sampling period: **{1/fs:.3f} seconds**
        - number of samples: **{len(t_samp)}**
    """)
    

with col2:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_cont, x_cont, color="black", label="Continuous sine")
    ax.stem(t_samp, x_samp, linefmt="r-", markerfmt="ro", basefmt=" ", label="Samples")
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sampling a 0.5 Hz sine wave")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig)


st.header("2. What is a system?")

st.markdown("""
A **system** transforms an input sequence $\{x[n]\}$ into an output sequence $\{y[n]\}$:

$\{y[n]\} = H\{x[n]\}$

We'll investigate some of the simplest systems we mentioned in class for you to investigate
what they do to the input signal.
""")

n = np.arange(-10, 11)
x_base = np.array([0, 1, 2, 3, 2, 1, 0, -1, -2, -1, 0, 1, 0, 2, 1, 0, 0, -1, 0, 1, 0], dtype=float)

col1, col2 = st.columns([0.3, 0.7])

with col1:
    system_choice = st.selectbox(
        "Select a system",
        [
            "Shift",
            "Flip",
            "Scale",
        ]
    )

    shift_k = 1
    scale_factor = 2

    if system_choice == "Shift":
        shift_k = st.slider("Shift by $k$ samples:", -20, 20, 0, 1)
        st.markdown("$y[n] = x[n-k]$")
    elif system_choice == "Scale":
        scale_factor = st.slider("Scale factor:", 1, 5, 2, 1)
        st.markdown("$y[n] = x[m\cdot n]$")
    elif system_choice == "Flip":
        st.markdown("$y[n] = x[-n]$")

    y_base = helpers.apply_system_by_name(
        x_base,
        system_choice,
        shift_k=shift_k,
        scale_factor=scale_factor,
    )

with col2:
    ylim = 1.15 * helpers.max_abs(x_base, y_base)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    helpers.stem_plot(axs[0], n, x_base, title="Input signal x[n]", ylim=(-ylim, ylim), color="black")
    helpers.stem_plot(axs[1], n, y_base, title=f"Output signal y[n] — {system_choice}", color="tab:red", ylim=(-ylim, ylim))
    axs[1].set_xlim(n[0] - 0.5, n[-1] + 0.5)
    plt.tight_layout()
    st.pyplot(fig)


st.header("3. What does superposition mean?")

st.markdown(r"""
A system is **linear** if it satisfies:

$H\{a x_1[n] + b x_2[n]\} = aH\{x_1[n]\} + bH\{x_2[n]\}$

Linear systems adhere to the **principle of superposition**: no matter if we 
- (1) sum the inputs, then (2) pass the sum through the system, or 
- (2) pass the individual signals through the system and then (1) sum up,
            
the results are identical!
""")

n_sup = np.arange(0, 21)
x1 = np.sin(2 * np.pi * 0.08 * n_sup)
x2 = np.where((n_sup >= 5) & (n_sup <= 12), 1.0, 0.0)

linear_test_system = st.selectbox(
    "System to test",
    [
        "Shift",
        "Flip",
        "Scale",
        "Square"
    ]
)



alpha = 1
beta = 1



if linear_test_system == "Shift":
    f = lambda sig: helpers.shift_signal_by_samples(sig, 3)
elif linear_test_system == "Flip":
    f = lambda sig: helpers.flip_signal(sig)
elif linear_test_system == "Scale":
    f = lambda sig: helpers.index_scale_signal(sig, 2)
else:
    f = lambda sig: sig ** 2

left, right, ok = helpers.test_superposition(f, x1, x2, alpha=alpha, beta=beta)


combo = alpha * x1 + beta * x2
ylim = 1.25 * helpers.max_abs(combo, left, right)

fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
helpers.stem_plot(axs[0], n_sup, combo, title=r"Combined input: $a x_1[n] + b x_2[n]$", ylim=(-ylim, ylim), color='black')
helpers.stem_plot(axs[1], n_sup, left, title=r"Left side: $H\{a x_1[n] + b x_2[n]\}$", ylim=(-ylim, ylim), color='black')
helpers.stem_plot(axs[2], n_sup, right, title=r"Right side: $aH\{x_1[n]\} + bH\{x_2[n]\}$", color="tab:red", ylim=(-ylim, ylim))
axs[2].set_xlim(n_sup[0] - 0.75, n_sup[-1] + 0.75)
plt.tight_layout()
st.pyplot(fig)

if ok:
    st.badge("**The two outputs match:** superposition holds for this example.", color="orange")
else:
    st.badge("**The two outputs do not match:** superposition does not hold here.", color="red")


st.header("4. Impulse response: how does a system react to a delta input?")

st.markdown(r"""
The **impulse response** of a system is its output when the input is a **delta function / impulse**:

$h[n] = H\{\delta[n]\}$

This is especially important for LTI systems, because once we know $h[n]$,
we can compute the output to **any** input signal by convolution.
""")

n_imp = np.linspace(-10,10, 21)
delta = helpers.delta_sequence(n_imp, 0)

col1, col2 = st.columns([0.3, 0.7])

with col1:
    impulse_system = st.selectbox(
        "Choose a system for delta input",
        [
            "Shift",
            "Flip",
            "Scale",
            "Moving average",
            "Forward difference"
        ]
    )

    if impulse_system == "Shift":
        shift_k = st.slider("Shift by $k$ samples:", -10, 10, 0, 1)
        h_out = helpers.shift_signal_by_samples(delta, shift_k)
        h_n = n_imp
    elif impulse_system == "Flip":
        h_out = helpers.flip_signal(delta)
        h_n = n_imp
    elif impulse_system == "Scale":
        scale_f = st.slider("Shift by $k$ samples:", 1, 4, 1, 1)
        h_out = helpers.index_scale_signal(delta, scale_f)
        h_n = n_imp
    elif impulse_system == "Moving average":
        avg_points = st.slider("Average over $n$ samples:", 1,5,3,1)
        h_out, h_kernel = helpers.moving_average_full(delta, avg_points)
        h_n = np.arange(n_imp[0], n_imp[0] + len(h_out))
        st.markdown("$y[n] = x[n-k+1]+\dots+x[n]$")
    elif impulse_system == "Forward difference":
        h_out, h_kernel = helpers.forward_difference_full(delta)
        h_out = h_out[::-1]
        h_out = h_out[1::]
        h_n = np.arange(n_imp[0], n_imp[0] + len(h_out))
        st.markdown("$y[n] = x[n+1]-x[n]$")

with col2:
    ylim = 1.25 * helpers.max_abs(delta, h_out)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    helpers.stem_plot(axs[0], n_imp, delta, title=r"Input: $\delta[n]$", ylim=(-ylim, ylim), color="black")
    helpers.stem_plot(axs[1], h_n, h_out, title=rf"Output / impulse response $h[n]$", color="tab:red", ylim=(-ylim, ylim))
    plt.tight_layout()
    st.pyplot(fig)



st.header("5. Convolution")

st.markdown(r"""
For an LTI system, the output is always a **convolution** of the **input** with the **system's 
impulse response**:

$y[n] = x[n] * h[n]$

            """)

col1, col2 = st.columns([0.3, 0.7])

with col1:
    st.markdown("""
    You can think of convolution as the following approach:

    1. take the impulse response $h[n]$
    2. slide it across the signal
    3. multiply overlapping values
    4. add them up
    """)

    x_text2 = st.text_input("x[n] for sliding view", "4, 3, 2, 3, 1", key="x2")
    h_text2 = st.text_input("h[n] for sliding view", "0.333, 0.333, 0.333", key="h2")

    try:
        x_viz = np.array([float(v.strip()) for v in x_text2.split(",")], dtype=float)
    except Exception:
        x_viz = np.array([1,2,3,3,3,2,1,3,1,5,1,1,2], dtype=float)
        st.warning("Could not parse x[n] for sliding view. Using default.")

    try:
        h_viz = np.array([float(v.strip()) for v in h_text2.split(",")], dtype=float)
    except Exception:
        h_viz = np.array([1/3, 1/3, 1/3], dtype=float)
        st.warning("Could not parse h[n] for sliding view. Using default.")

    y_viz = np.convolve(x_viz, h_viz, mode="full")

    N = len(x_viz) + len(h_viz) - 1
    common_n = np.arange(N)

    out_idx = st.slider("Output index n₀", 0, len(y_viz) - 1, 0, 1)

    x_pad = np.zeros(N)
    x_pad[:len(x_viz)] = x_viz

    h_shifted = helpers.shift_kernel_for_convolution(h_viz, out_idx, N)
    product = x_pad * h_shifted

    st.latex(rf"y[{out_idx}] = {y_viz[out_idx]:.3f}")

with col2:
    ylim = 1.15 * helpers.max_abs(x_pad, h_shifted, product, y_viz)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    helpers.stem_plot(
        axs[0], common_n, x_pad,
        title="Input $x[k]$",
        ylim=(-ylim, ylim),
        color="black"
    )

    helpers.stem_plot(
        axs[1], common_n, h_shifted,
        title=rf"Shifted kernel $h[{out_idx}-k]$",
        color="tab:orange",
        ylim=(-0.25, 1.25)
    )

    helpers.stem_plot(
        axs[2], common_n, product,
        title=r"Pointwise product $x[k]\cdot h[n_0-k]$",
        color="black",
        ylim=(-ylim, ylim)
    )

    helpers.stem_plot(
        axs[3], common_n, y_viz,
        title=r"Output $y[n]$",
        color="tab:red",
        ylim=(-ylim, ylim)
    )

    axs[3].axvline(out_idx, linestyle="--", color="black")
    axs[3].set_xlim(common_n[0] - 0.5, common_n[-1] + 0.5)

    plt.tight_layout()
    st.pyplot(fig)
