from matplotlib.pylab import grid
from scipy.constants import g
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Lecture 4 - Digital Filtering (Part 1)",
    layout="wide",
)

st.title("Lecture 4: Digital Filtering, first part")

st.header("1. Ideal filters")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        r"""
        Here we look at the **ideal magnitude responses** of common filters in the
        **both-sided spectrum**.

        We assume an ideal discrete-time frequency response magnitude $|H(f)|$
        over negative and positive frequencies.
        """
    )

    filter_type = st.selectbox(
        "Filter type",
        ["Lowpass", "Highpass", "Bandpass", "Bandstop", "Allpass"]
    )

    fmax = 10.0

    freqs = np.linspace(-fmax, fmax, 2001)
    H = np.zeros_like(freqs)

    if filter_type == "Lowpass":
        fc = st.slider("Cutoff frequency $f_c$", 0.1, fmax, 3.0, step=0.1)
        H[np.abs(freqs) <= fc] = 1.0

    elif filter_type == "Highpass":
        fc = st.slider("Cutoff frequency $f_c$", 0.1, fmax, 3.0, step=0.1)
        H[np.abs(freqs) >= fc] = 1.0

    elif filter_type == "Bandpass":
        f1 = st.slider("Lower cutoff $f_{c1}$", 0.1, fmax - 0.1, 2.0, step=0.1)
        f2 = st.slider("Upper cutoff $f_{c2}$", f1 + 0.1, fmax, 5.0, step=0.1)
        H[(np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)] = 1.0

    elif filter_type == "Bandstop":
        f1 = st.slider("Lower cutoff $f_{c1}$", 0.1, fmax - 0.1, 2.0, step=0.1)
        f2 = st.slider("Upper cutoff $f_{c2}$", f1 + 0.1, fmax, 5.0, step=0.1)
        H[:] = 1.0
        H[(np.abs(freqs) >= f1) & (np.abs(freqs) <= f2)] = 0.0

    elif filter_type == "Allpass":
        H[:] = 1.0

with col2: 
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs,
        y=H,
        mode="lines",
        line=dict(color="red", width=3, shape="hv"),
        name="|H(f)|"
    ))

    fig.add_trace(go.Scatter(
        x=freqs,
        y=1 - H,
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(255, 0, 0, 0.18)",
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_hline(y=1, line=dict(color="gray", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="black", width=1))

    fig.update_layout(
        title=f"Ideal {filter_type.lower()} magnitude response",
        xaxis_title="Frequency",
        yaxis_title="Magnitude |H(f)|",
        template="plotly_white",
        height=400,
        showlegend=False
    )

    fig.update_yaxes(range=[-0.1, 1.1])

    st.plotly_chart(fig, width='content')



st.header("2. Filtering in the frequency domain")
st.markdown(
    r"""
    We now take a noisy signal, compute its DFT, and compare its **original
    spectrum** with the spectrum **after multiplication by an ideal filter's response**.
    """
)

fs_demo = 100.0
N_demo = 2048
t_demo = np.arange(N_demo) / fs_demo

rng = np.random.default_rng(42)

freqs = np.fft.fftfreq(N_demo, d=1/fs_demo)

mag_shape = np.zeros_like(freqs)
nonzero = freqs != 0
mag_shape[nonzero] = 1 / np.sqrt(np.abs(freqs[nonzero]))  # ~ 1/f^(1/2) in magnitude → 1/f power

random_phase = np.exp(1j * 2 * np.pi * rng.random(N_demo))
X_pink = mag_shape * random_phase

X_pink[0] = 0
X_pink = np.fft.ifftshift(X_pink)
x_demo = np.fft.ifft(X_pink).real

x_demo = x_demo / np.std(x_demo)
X_demo = np.fft.fft(x_demo)
f_demo = np.fft.fftfreq(N_demo, d=1 / fs_demo)

X_demo_shift = np.fft.fftshift(X_demo)
f_demo_shift = np.fft.fftshift(f_demo)

mag_demo = np.abs(X_demo_shift)


fig_time_demo = go.Figure()

fig_time_demo.add_trace(go.Scatter(
    x=t_demo,
    y=x_demo,
    mode="lines",
    name="Signal",
    line=dict(color="black", width=1.5)
))

fig_time_demo.update_layout(
    title="Signal in time domain",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=280,
    showlegend=False
)

st.plotly_chart(fig_time_demo, width='stretch')


col1, col2 = st.columns([0.35, 0.65])

with col1:
    filter_type_2 = st.selectbox(
        "Filter type",
        ["Lowpass", "Highpass", "Bandpass", "Bandstop", "Allpass"],
        key="filter_type_section_2"
    )

    fmax_demo = 10.0
    H_demo = np.zeros_like(f_demo_shift)

    if filter_type_2 == "Lowpass":
        fc_2 = st.slider(
            "Cutoff frequency fc (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo),
            value=8.0,
            step=0.1,
            key="fc_section_2_low"
        )
        H_demo[np.abs(f_demo_shift) <= fc_2] = 1.0

    elif filter_type_2 == "Highpass":
        fc_2 = st.slider(
            "Cutoff frequency fc (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo),
            value=8.0,
            step=0.1,
            key="fc_section_2_high"
        )
        H_demo[np.abs(f_demo_shift) >= fc_2] = 1.0

    elif filter_type_2 == "Bandpass":
        f1_2 = st.slider(
            "Lower cutoff f1 (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo - 0.2),
            value=2.0,
            step=0.1,
            key="f1_section_2_bp"
        )
        f2_2 = st.slider(
            "Upper cutoff f2 (Hz)",
            min_value=float(f1_2 + 0.1),
            max_value=float(fmax_demo),
            value=10.0,
            step=0.1,
            key="f2_section_2_bp"
        )
        H_demo[(np.abs(f_demo_shift) >= f1_2) & (np.abs(f_demo_shift) <= f2_2)] = 1.0

    elif filter_type_2 == "Bandstop":
        f1_2 = st.slider(
            "Lower cutoff f1 (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo - 0.2),
            value=2.0,
            step=0.1,
            key="f1_section_2_bs"
        )
        f2_2 = st.slider(
            "Upper cutoff f2 (Hz)",
            min_value=float(f1_2 + 0.1),
            max_value=float(fmax_demo),
            value=10.0,
            step=0.1,
            key="f2_section_2_bs"
        )
        H_demo[:] = 1.0
        H_demo[(np.abs(f_demo_shift) >= f1_2) & (np.abs(f_demo_shift) <= f2_2)] = 0.0

    elif filter_type_2 == "Allpass":
        H_demo[:] = 1.0

# Multiply spectrum by ideal filter
X_filt_demo_shift = X_demo_shift * H_demo
mag_filt_demo = np.abs(X_filt_demo_shift)

with col2:
    fig_spec_demo2 = go.Figure()

    # original spectrum
    fig_spec_demo2.add_trace(go.Scatter(
        x=f_demo_shift,
        y=mag_demo,
        mode="lines",
        name="Original spectrum X[k]",
        line=dict(color="black", width=1.8)
    ))

    # stopband shading
    fig_spec_demo2.add_trace(go.Scatter(
        x=f_demo_shift,
        y=(1 - H_demo) * np.max(mag_demo),
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.14)",
        showlegend=False,
        hoverinfo="skip"
    ))

    # filtered spectrum
    fig_spec_demo2.add_trace(go.Scatter(
        x=f_demo_shift,
        y=mag_filt_demo,
        mode="lines",
        name=r"|X[k]H[k]|",
        line=dict(color="red", width=2.2)
    ))

    fig_spec_demo2.update_layout(
        title=f"Both-sided magnitude spectrum with ideal {filter_type_2.lower()} filtering",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
        height=420,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )

    fig_spec_demo2.update_xaxes(range=[-20, 20])
    fig_spec_demo2.update_yaxes(range=[-20, 40])

    st.plotly_chart(fig_spec_demo2, width='stretch')



st.header("3. Filtering a sum of sinusoids")
st.markdown(
    r"""
    We now take a signal that is the sum of three sinusoids. We compute its FFT,
    multiply the spectrum by the selected ideal filter, and then transform back
    to the time domain using the inverse FFT.
    """
)


fs_demo3 = 100.0
N_demo3 = 2048
t_demo3 = np.arange(N_demo3) / fs_demo3

x_demo3 = (
    1.0 * np.sin(2 * np.pi * 2.0 * t_demo3)
    + 0.7 * np.sin(2 * np.pi * 6.0 * t_demo3 + 0.6)
    + 0.45 * np.sin(2 * np.pi * 12.0 * t_demo3 + 1.1)
)


X_demo3 = np.fft.fft(x_demo3)
f_demo3 = np.fft.fftfreq(N_demo3, d=1 / fs_demo3)

X_demo3_shift = np.fft.fftshift(X_demo3)
f_demo3_shift = np.fft.fftshift(f_demo3)

mag_demo3 = np.abs(X_demo3_shift)


col1, col2 = st.columns([0.32, 0.68])

with col1:
    filter_type_3 = st.selectbox(
        "Filter type",
        ["Lowpass", "Highpass", "Bandpass", "Bandstop", "Allpass"],
        key="filter_type_section_3"
    )

    fmax_demo3 = 14.0
    H_demo3 = np.zeros_like(f_demo3_shift)

    if filter_type_3 == "Lowpass":
        fc_3 = st.slider(
            "Cutoff frequency fc (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo3),
            value=4.0,
            step=0.1,
            key="fc_section_3_low"
        )
        H_demo3[np.abs(f_demo3_shift) <= fc_3] = 1.0

    elif filter_type_3 == "Highpass":
        fc_3 = st.slider(
            "Cutoff frequency fc (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo3),
            value=4.0,
            step=0.1,
            key="fc_section_3_high"
        )
        H_demo3[np.abs(f_demo3_shift) >= fc_3] = 1.0

    elif filter_type_3 == "Bandpass":
        f1_3 = st.slider(
            "Lower cutoff f1 (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo3 - 0.2),
            value=4.0,
            step=0.1,
            key="f1_section_3_bp"
        )
        f2_3 = st.slider(
            "Upper cutoff f2 (Hz)",
            min_value=float(f1_3 + 0.1),
            max_value=float(fmax_demo3),
            value=8.0,
            step=0.1,
            key="f2_section_3_bp"
        )
        H_demo3[(np.abs(f_demo3_shift) >= f1_3) & (np.abs(f_demo3_shift) <= f2_3)] = 1.0

    elif filter_type_3 == "Bandstop":
        f1_3 = st.slider(
            "Lower cutoff f1 (Hz)",
            min_value=0.1,
            max_value=float(fmax_demo3 - 0.2),
            value=4.0,
            step=0.1,
            key="f1_section_3_bs"
        )
        f2_3 = st.slider(
            "Upper cutoff f2 (Hz)",
            min_value=float(f1_3 + 0.1),
            max_value=float(fmax_demo3),
            value=8.0,
            step=0.1,
            key="f2_section_3_bs"
        )
        H_demo3[:] = 1.0
        H_demo3[(np.abs(f_demo3_shift) >= f1_3) & (np.abs(f_demo3_shift) <= f2_3)] = 0.0

    elif filter_type_3 == "Allpass":
        H_demo3[:] = 1.0


Y_demo3_shift = X_demo3_shift * H_demo3
mag_filt_demo3 = np.abs(Y_demo3_shift)

# undo fftshift before ifft
Y_demo3 = np.fft.ifftshift(Y_demo3_shift)
y_demo3 = np.fft.ifft(Y_demo3).real


with col2:
    fig_spec_demo3 = go.Figure()

    fig_spec_demo3.add_trace(go.Scatter(
        x=f_demo3_shift,
        y=mag_demo3,
        mode="lines",
        name="Original spectrum",
        line=dict(color="black", width=1.6)
    ))

    fig_spec_demo3.add_trace(go.Scatter(
        x=f_demo3_shift,
        y=(1 - H_demo3) * np.max(mag_demo3),
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.14)",
        showlegend=False,
        hoverinfo="skip"
    ))

    fig_spec_demo3.add_trace(go.Scatter(
        x=f_demo3_shift,
        y=mag_filt_demo3,
        mode="lines",
        name="Filtered spectrum",
        line=dict(color="red", width=2.2)
    ))

    fig_spec_demo3.update_layout(
        title=f"Both-sided magnitude spectrum with ideal {filter_type_3.lower()} filtering",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
        height=380,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )

    fig_spec_demo3.update_xaxes(range=[-20, 20])

    st.plotly_chart(fig_spec_demo3, width='stretch')

fig_time_demo3 = go.Figure()

fig_time_demo3.add_trace(go.Scatter(
    x=t_demo3,
    y=x_demo3,
    mode="lines",
    name="Original signal",
    line=dict(color="black", width=1.5)
))

fig_time_demo3.add_trace(go.Scatter(
    x=t_demo3,
    y=y_demo3,
    mode="lines",
    name="Filtered signal",
    line=dict(color="red", width=2.0)
))

fig_time_demo3.update_layout(
    title="Time-domain signals before and after filtering",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=320,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.35,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80)
)

fig_time_demo3.update_xaxes(range=[0, 3])
st.plotly_chart(fig_time_demo3, width='stretch')




st.subheader("4. Artifacts of ideal filtering: ringing and the Gibbs phenomenon")
st.subheader("4.1 Ringing")
st.markdown(
    r"""
    Sharp transitions in a signal require many sinusoidal components to be represented.
    If we lowpass-filter such a signal, some of these high-frequency components are removed.
    This causes oscillatory overshoots and undershoots near the edges: **ringing**.
    Windowing the impulse response reduces ringing, but makes the frequency response less sharp.
    """
)

fs_ring = 200.0
N_ring = 2048
t_ring = np.arange(N_ring) / fs_ring
f_ring = np.fft.fftfreq(N_ring, d=1 / fs_ring)
f_ring_shift = np.fft.fftshift(f_ring)

x_ring = np.zeros(N_ring)

def sec_to_idx(sec):
    return int(sec * fs_ring)

x_ring[sec_to_idx(1.0):sec_to_idx(2.5)] = 1.0
x_ring[sec_to_idx(4.0):sec_to_idx(5.5)] = 1.0
x_ring[sec_to_idx(7.0):sec_to_idx(8.5)] = 1.0

col1, col2 = st.columns(2)

with col1:
    fc_ring = st.slider(
        "Lowpass cutoff frequency fc (Hz)",
        min_value=0.5,
        max_value=float(fs_ring / 2),
        value=8.0,
        step=0.5,
        key="ringing_step_fc"
    )

with col2:
    window_type_ring = st.selectbox(
        "Window",
        ["Rectangular", "Hann", "Hamming", "Blackman"],
        key="ringing_step_window"
    )

H_ideal = np.zeros(N_ring)
H_ideal[np.abs(f_ring) <= fc_ring] = 1.0
H_ideal_shift = np.fft.fftshift(H_ideal)

# Ideal impulse response
h_ideal = np.fft.ifft(H_ideal).real
h_ideal_plot = np.fft.fftshift(h_ideal)

def make_window(window_type: str, N: int) -> np.ndarray:
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

L = 151

w_short = make_window(window_type_ring, L)

w_ring = np.zeros(N_ring)
center = N_ring // 2
half = L // 2
w_ring[center - half:center + half + 1] = w_short

# Window the centered impulse response
h_win_plot = h_ideal_plot * w_ring

# Shift back to FFT order for transform/filtering
h_win = np.fft.ifftshift(h_win_plot)

# Frequency response of the windowed filter
H_win = np.fft.fft(h_win)
H_win_shift = np.fft.fftshift(H_win)

# Filter the rectangular signal
X_ring = np.fft.fft(x_ring)
Y_ring = X_ring * H_win
y_ring = np.fft.ifft(Y_ring).real

with col1:
    fig_freq = go.Figure()

    fig_freq.add_trace(go.Scatter(
        x=f_ring_shift,
        y=np.abs(H_win_shift),
        mode="lines",
        name="Windowed filter response",
        line=dict(color="red", width=1.5)
    ))

    fig_freq.add_trace(go.Scatter(
        x=f_ring_shift,
        y=np.abs(H_ideal_shift),
        mode="lines",
        name="Ideal lowpass",
        line=dict(color="black", width=1.5)
    ))

    fig_freq.update_layout(
        title=f"Frequency response ({window_type_ring} window)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
        height=320,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )

    fig_freq.update_xaxes(range=[-25, 25])

    st.plotly_chart(fig_freq, use_container_width=True)

with col2:
    n = np.arange(N_ring)
    n_center = n - N_ring // 2
    t_h = n_center / fs_ring

    zoom_half = 150
    idx0 = N_ring // 2 - zoom_half
    idx1 = N_ring // 2 + zoom_half

    fig_imp = go.Figure()

    fig_imp.add_trace(go.Scatter(
        x=t_h[idx0:idx1],
        y=h_win_plot[idx0:idx1],
        mode="lines",
        name=f"{window_type_ring} windowed",
        line=dict(color="red", width=1.5)
    ))

    fig_imp.add_trace(go.Scatter(
        x=t_h[idx0:idx1],
        y=h_ideal_plot[idx0:idx1],
        mode="lines",
        name="Ideal impulse response",
        line=dict(color="black", width=1.5)
    ))

    fig_imp.update_layout(
        title="Impulse response of the filter",
        xaxis_title="Time lag (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=320,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=80)
    )

    st.plotly_chart(fig_imp, use_container_width=True)

fig_step = go.Figure()

fig_step.add_trace(go.Scatter(
    x=t_ring,
    y=x_ring,
    mode="lines",
    name="Input",
    line=dict(color="black", width=2)
))

fig_step.update_layout(
    title="Input signal: rectangular signal with several edges",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=260,
    showlegend=False
)

st.plotly_chart(fig_step, use_container_width=True)

Y_ideal = X_ring * H_ideal
y_ideal = np.fft.ifft(Y_ideal).real

fig_out = go.Figure()

fig_out.add_trace(go.Scatter(
    x=t_ring,
    y=y_ideal,
    mode="lines",
    name="Without window",
    line=dict(color="black", width=1.5)
))

fig_out.add_trace(go.Scatter(
    x=t_ring,
    y=y_ring,
    mode="lines",
    name=f"With {window_type_ring} window",
    line=dict(color="red", width=1.5)
))

fig_out.update_layout(
    title="Filtered signal: reconstruction with and without window",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=320,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.4,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80)
)

fig_out.update_xaxes(range=[0, 9])

st.plotly_chart(fig_out, use_container_width=True)



st.subheader("4.2 Temporal aliasing")

st.markdown(r"**a) One perspective: Downsampling**")

st.markdown(
    r"""
    We've seen before that changes in time domain without care (i.e., downsampling without
    prior lowpass filtering) leads to **frequency domain aliasing** (commonly just called
    **aliasing**). 

    The thing is, though: the same holds the other way around. If we tinker with the 
    spectrum, then do an iDFT, we likewise introduce artifacts in the temporal domain that,
    once again, relate to the assumed periodicity of the signals in the DFT and iDFT.

    Let's take a look at "downsampling" in frequency domain (doesn't make much sense, but
    bear with me to show you that the effect is **exactly** the same as what we've seen
    before).
    """
)

fs_alias = 200.0
N_alias = 2000
t_alias = np.arange(N_alias) / fs_alias
T_alias = t_alias[-1]

# same signal as before: increasing-amplitude sinusoid
f0_alias = 3.0
x_alias = np.sin(2 * np.pi * f0_alias * t_alias) * (t_alias / T_alias)
x_alias[200] = 1
x_alias[1100] = -1
x_alias[1400] = -2

# FFT
X_alias = np.fft.fft(x_alias)
f_alias = np.fft.fftfreq(N_alias, d=1 / fs_alias)
f_alias_shift = np.fft.fftshift(f_alias)
X_alias_shift = np.fft.fftshift(X_alias)


M_alias = st.selectbox(
    "Keep every M-th spectral sample",
    [2, 3, 4, 5, 6, 8, 12, 16],
    index=0,
    key="freq_domain_downsampling_factor"
)

# keep same total length, zero out everything else
X_ds_shift = X_alias_shift.copy()
#X_ds_shift[::M_alias] = 0

# back to FFT order for inverse transform
X_ds = np.fft.ifftshift(X_ds_shift[::M_alias])
x_ds = np.fft.ifft(X_ds).real


fig_alias_spec = go.Figure()

fig_alias_spec.add_trace(go.Scatter(
    x=f_alias_shift,
    y=np.abs(X_alias_shift),
    mode="lines",
    name="Original spectrum",
    line=dict(color="black", width=1.5)
))

fig_alias_spec.add_trace(go.Scatter(
    x=f_alias_shift,
    y=np.abs(X_ds_shift),
    mode="lines",
    name=f"Every {M_alias}-th spectral sample kept",
    line=dict(color="red", width=1.5)
))

fig_alias_spec.update_layout(
    title=f"Spectrum after keeping every {M_alias}-th FFT sample",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
    template="plotly_white",
    height=320,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.35,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80)
)

fig_alias_spec.update_xaxes(range=[-40, 40])

st.plotly_chart(fig_alias_spec, use_container_width=True)

fig_alias_time = go.Figure()

fig_alias_time.add_trace(go.Scatter(
    x=t_alias,
    y=x_alias,
    mode="lines",
    name="Original signal",
    line=dict(color="black", width=1.5)
))

fig_alias_time.add_trace(go.Scatter(
    x=t_alias,
    y=x_ds,
    mode="lines",
    name=f"After keeping every {M_alias}-th spectral sample",
    line=dict(color="red", width=1.5)
))

fig_alias_time.update_layout(
    title="Time domain: aliasing / periodic replication after spectral downsampling",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=340,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.4,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80)
)

st.plotly_chart(fig_alias_time, use_container_width=True)
st.markdown(r"""
    **To summarize:**
    - downsampling in frequency domain means we reduce the frequency bins we regard
    - from initially $N$ bins, we suddenly only retain $M=N/2$ 
    - if we now backtransform, the time domain signal also only has $M$ samples anymore!
    - the issue, though: the initial signal was way longer $\rightarrow$ the surplus in
    signal is now assumed as part of a novel 'period' in time domain
    - we see the initial signal overlapping into the novel iDFT period $\rightarrow$
    this is **temporal domain aliasing**
""")


st.markdown(r"**b) Another viewpoint: Circular convolution**")
st.markdown(
    r"""
    We said multiplication in one domain corresponds to circular (!) convolution in the
    other domain - this is what actually induces aliasing, both in frequency and in time
    domain. 

    We can regard this with an M-point moving average filter. Its impulse response (as
    we calculated in the first lecture) is just a window of ones with M entries, 
    normalized by M. Convolving with this impulse response in time domain means that, for
    an M-point moving average, (M-1) values of the edge are taken from the other edge -
    our information of one end spills over into the other and vice verse!
    $\rightarrow$ **temporal domain aliasing**
    """
)

fs = 200.0
N = 500
time = np.arange(N) / fs
T = time[-1]*5

# test signal: increasing sinusoid
f0 = 15.0
signal = np.sin(2 * np.pi * f0 * time) * (time / T)

M = st.slider(
    "Moving-average length M",
    min_value=1,
    max_value=20,
    value=1,
    step=2,
    key="temp_alias_mavg_len"
)

h = np.ones(M) / M

# center it → zero-phase
h_zero = np.zeros(N)
half = M // 2
h_zero[N//2 - half : N//2 - half + M] = h

# shift back to FFT order
h_zero = np.fft.ifftshift(h_zero)

H = np.fft.fft(h_zero)
H = H / np.max(np.abs(H))
Signal = np.fft.fft(signal, n=N)
Y_circ = Signal * H
y_circ = np.fft.ifft(Y_circ).real

# Frequency axis for plotting
freqs = np.fft.fftfreq(N, d=1 / fs)
f_shift = np.fft.fftshift(freqs)
X_shift = np.fft.fftshift(Signal)
H_shift = np.fft.fftshift(H)
Y_shift = np.fft.fftshift(Y_circ)


fig_alias_spec = go.Figure()

fig_alias_spec.add_trace(go.Scatter(
    x=f_shift,
    y=np.abs(X_shift),
    mode="lines",
    name="Signal spectrum",
    line=dict(color="black", width=1.5)
))

fig_alias_spec.add_trace(go.Scatter(
    x=f_shift,
    y=np.abs(H_shift) * np.max(np.abs(X_shift)),
    mode="lines",
    name="Moving-average response (scaled)",
    line=dict(color="gray", width=1.5, dash="dash")
))

fig_alias_spec.add_trace(go.Scatter(
    x=f_shift,
    y=np.abs(Y_shift),
    mode="lines",
    name="Filtered spectrum",
    line=dict(color="red", width=1.8)
))

fig_alias_spec.update_layout(
    title=f"Spectrum multiplication with the DFT of an {M}-point moving average",
    xaxis_title="Frequency (Hz)",
    yaxis_title="Magnitude",
    template="plotly_white",
    height=320,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.35,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80)
)

fig_alias_spec.update_xaxes(range=[-20, 20])

st.plotly_chart(fig_alias_spec, use_container_width=True)


fig_alias_time = go.Figure()

fig_alias_time.add_trace(go.Scatter(
    x=time,
    y=signal,
    mode="lines",
    name="Original signal",
    line=dict(color="black", width=1.5)
))

fig_alias_time.add_trace(go.Scatter(
    x=time,
    y=y_circ,
    mode="lines",
    name="Circular convolution (FFT multiplication)",
    line=dict(color="red", width=2)
))

fig_alias_time.update_layout(
    title="Time domain: wrap-around due to circular convolution",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude",
    template="plotly_white",
    height=360,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.4,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=80)
)

st.plotly_chart(fig_alias_time, use_container_width=True)


alias_len = M - 1
t_alias = np.arange(alias_len) / fs

