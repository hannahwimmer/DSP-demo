import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import src.helpers as helpers
import io
from scipy.io import wavfile
import numpy as np
import scipy.signal as scisig
import scipy.signal as sig
import plotly.express as px

st.set_page_config(
    page_title="Lecture - Time-Frequency Analysis and Filtering",
    layout="wide",
)

st.title("Lecture 3: Further analyses and filtering")

st.header("1. Analyzing a signal's frequency content")
st.markdown(
    r"""
    We already discussed how to analyze a signal's frequency content via the DFT.
    Let's do that quickly for an audio we record now and take a look at the magnitude
    spectrum:
"""
)
recording = st.audio_input("Record audio", sample_rate=44100)

st.markdown("""
    

""")

if recording is not None:
    audio_bytes = recording.read()
    fs, signal = wavfile.read(io.BytesIO(audio_bytes))
    st.write(f"Sampling rate: {fs}Hz")

    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max
    else:
        signal = signal.astype(np.float64)

    # stereo → mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    freqs, magnitude, power, X = helpers.one_sided_spectrum(signal, fs=fs)

    f_stft, t_stft, Zxx = scisig.stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        padded=True
    )

    stft_db = 20 * np.log10(np.abs(Zxx) + 1e-8)

    time_signal = np.linspace(0, (len(signal)-1)/fs, len(signal))

    exp1 = st.expander("See signal")
    with exp1:
        fig_time = go.Figure()

        fig_time.add_trace(go.Scatter(
            x=time_signal,
            y=signal,
            mode="lines",
            line=dict(color="black"),
            name="Signal",
        ))

        fig_time.update_layout(
            title="Recorded signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=250,
            showlegend=False
        )

        st.plotly_chart(fig_time, use_container_width=True)

    exp2 = st.expander("See spectrum")
    with exp2:
        fig_spec = go.Figure()

        fig_spec.add_trace(go.Scatter(
            x=freqs,
            y=magnitude,
            mode="lines",
            line=dict(color="orange"),
            name="Magnitude",
        ))

        tickbox = st.checkbox("Show pitch (A major scale)")

        if tickbox:
            base = 220
            semitones = [0, 2, 4, 5, 7, 9, 11, 12]
            semitone_names = ["a", "h", "c#'", "d'", "e'", "f#'", "g#'", "a'"]
            y_text = magnitude.max() * 0.8

            for s, name in zip(semitones, semitone_names):
                f_line = base * 2**(s/12)

                fig_spec.add_trace(go.Scatter(
                    x=[f_line, f_line],
                    y=[0, magnitude.max()],
                    mode="lines",
                    line=dict(dash="dot", color="black"),
                    showlegend=False
                ))

                fig_spec.add_annotation(
                    x=f_line,
                    y=y_text,
                    text=name,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom"
                )

        fig_spec.update_layout(
            title="DFT Spectrum of the recording",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            template="plotly_white",
            height=300,
            showlegend=False
        )
        fig_spec.update_xaxes(range=[0,1000])

        st.plotly_chart(fig_spec, use_container_width=True)

    exp3 = st.expander("See spectrogram")
    with exp3:
        fig_stft = go.Figure()

        fig_stft.add_trace(go.Heatmap(
            z=stft_db,
            x=t_stft,
            y=f_stft,
            colorscale="Cividis",
            zmin=-70,
            zmax=0,
            colorbar=dict(title="dB"),
        ))

        fig_stft.update_layout(
            title="STFT Spectrogram of the recording",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            height=350,
        )

        st.plotly_chart(fig_stft, use_container_width=True)


st.header("2. The STFT and window sizes")


fs = 20

col1, col2 = st.columns(2)
with col1:
    window_type_label = st.selectbox(
    "Window type",
    ["Rectangular", "Hamming", "Hann", "Blackman"]
    )
    window_map = {
        "Rectangular": "boxcar",
        "Hamming": "hamming",
        "Hann": "hann",
        "Blackman": "blackman",
    }

    window_type = window_map[window_type_label]

    N = st.slider("Window size N", 4, 128, 32, step=2)
    M = st.slider("Overlap M", 0, N-1, N//2)

with col2:
    vals = np.linspace(0, 6, int(fs*6)+1)
    sins = np.sin(2*np.pi*6*vals/6)
    x = np.concatenate([np.zeros(int(7*fs)), sins, np.zeros(int(7*fs))])
    t_sig = np.arange(len(x))/fs


    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(x=t_sig, y=x, mode="lines", line=dict(color="black")))
    fig_sig.update_layout(title="Signal", height=250)
    st.plotly_chart(fig_sig, width='stretch')

    f, t, Zxx = sig.stft(
        x,
        fs=fs,
        window=window_type,
        nperseg=N,
        noverlap=M,
        nfft=N,
        return_onesided=False,
        boundary=None,
        padded=True
    )

    mag = np.abs(Zxx)
    Zplot = mag
    idx = np.argsort(f)
    f_plot = f[idx]
    Zplot_sorted = Zplot[idx, :]


col1, col2 = st.columns([0.6,0.4])
with col1:
    fig = go.Figure(data=go.Heatmap(
        x=t,
        y=f_plot,
        z=Zplot_sorted,
        colorscale="Cividis",
        colorbar=dict(title="|X[k,n]|")
    ))

    fig.update_layout(
        title=f"STFT ({window_type_label}, N={N}, M={M})",
        xaxis_title="time (s)",
        yaxis_title="frequency (Hz)",
        height=450
    )

    st.plotly_chart(fig, width='stretch')

with col2:
    fig_w = go.Figure()

    for i in range(len(t)):
        fig_w.add_trace(go.Scatter3d(
            x=f,
            y=np.full_like(f, t[i]),
            z=Zplot[:, i],
            mode="lines",
            line=dict(
                color=Zplot[:, i], 
                colorscale="Cividis",
                width=4
            ),
            showlegend=False
        ))

    fig_w.update_layout(
        title="Waterfall",
        scene=dict(
            xaxis_title="frequency (Hz)",
            yaxis_title="time (s)",
            zaxis_title="|X[k,n]|"
        ),
        height=450
    )

    st.plotly_chart(fig_w, width='stretch')


st.header("3. Window functions")

def make_window(window_type: str, grid_len: int, N: int) -> np.ndarray:
    if window_type == "Rectangular":
        window = np.zeros((grid_len, ))
        window[0:N] = 1
        return window
    elif window_type == "Hann":
        window = np.zeros((grid_len, ))
        window[0:N] = np.hanning(N)
        return window
    elif window_type == "Hamming":
        window = np.zeros((grid_len, ))
        window[0:N] = np.hamming(N)
        return window
    elif window_type == "Blackman":
        window = np.zeros((grid_len, ))
        window[0:N] = np.blackman(N)
        return window
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def shifted_window_on_grid(window: np.ndarray, grid_len: int, shift: int) -> np.ndarray:
    y = np.zeros(grid_len)
    N = len(window)

    start = shift
    end = shift + N

    vis_start = max(0, start)
    vis_end = min(grid_len, end)

    if vis_start < vis_end:
        win_start = vis_start - start
        win_end = win_start + (vis_end - vis_start)
        y[vis_start:vis_end] = window[win_start:win_end]

    return y


col1, col2, col3 = st.columns(3)

with col1:
    window_type = st.selectbox(
        "Window type",
        ["Rectangular", "Hann", "Hamming", "Blackman"]
    )

with col2:
    N = st.slider("Window length N", 8, 128, 32)

with col3:
    grid_len = st.slider("Display length", 32, 256, 96)

shift = st.slider("Shift m", -grid_len // 2, grid_len, 20)

window = make_window(window_type, grid_len, N)
shifted = shifted_window_on_grid(window, grid_len, shift)

n_grid = np.arange(grid_len)

window_equations = {
    "Rectangular": r"""
        w[n] =
        \begin{cases}
        1 & 0 \le n < M \\
        0 & \text{otherwise}
        \end{cases}
""",

    "Hann": r"""
        w[n] =
        \begin{cases}
        0.5-0.5\cdot cos(2\pi n/M) & 0 \le n < M \\
        0 & \text{otherwise}
        \end{cases}
""",

    "Hamming": r"""
        w[n] =
        \begin{cases}
        0.54-0.46\cdot cos(2\pi n/M) & 0 \le n < M \\
        0 & \text{otherwise}
        \end{cases}
""",

    "Blackman": r"""
        w[n] =
        \begin{cases}
        0.42-0.5\cdot cos(2\pi n/M) + 0.08 \cdot cos(4\pi n/M) & 0 \le n < M \\
        0 & \text{otherwise}
        \end{cases}
"""
}

st.markdown(f"""
**{window_type} window function:** 
$${window_equations[window_type]}$$
""")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=n_grid,
        y=window,
        mode="lines+markers",
        name="w[n]",
        line=dict(color="black"),
    ))

    # Shifted window
    fig.add_trace(go.Scatter(
        x=n_grid,
        y=shifted,
        mode="lines+markers",
        name=f"w[n-{shift}]",
        line=dict(color="red"),
    ))

    fig.update_layout(
        title="Window and Shifted Window",
        xaxis_title="n",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    n = np.arange(grid_len)
    x = np.cos(2 * np.pi * 5 * n/grid_len)
    x_m = x * shifted

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=n,
        y=x,
        mode="lines",
        name="x[n]",
        line=dict(dash="dash", color="lightgray"),
    ))

    fig2.add_trace(go.Scatter(
        x=n,
        y=shifted,
        mode="lines",
        name=f"w[n-{shift}]",
        line=dict(color="red"),
    ))

    fig2.add_trace(go.Scatter(
        x=n,
        y=x_m,
        mode="lines",
        name="x[n] · w[n-m]",
        line=dict(color="black"),
    ))

    fig2.update_layout(
        title="Signal, Window, and Windowed Signal",
        xaxis_title="n",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(fig2, width='stretch')


st.header("4. Windowing and spectral spreading")


col1, col2, col3 = st.columns(3)

with col1:
    spec_window_type = st.selectbox(
        "Window type for spectrum demo",
        ["Rectangular", "Hann", "Hamming", "Blackman"],
        key="spec_window_type"
    )

with col2:
    spec_N = st.slider(
        "Window length (samples)",
        min_value=20,
        max_value=600,
        value=120,
        step=2,
        key="spec_window_length"
    )

with col3:
    show_db = st.checkbox("Show spectrum in dB", value=False, key="spec_show_db")


fs_spec = 100
T_spec = 10.0
f0 = 2.0

t_spec = np.arange(0, T_spec, 1 / fs_spec)
x_spec = np.sin(2 * np.pi * f0 * t_spec)
total_len = len(x_spec)


window_start_time = 2.0
start_idx = int(window_start_time * fs_spec)
spec_N = min(spec_N, total_len - start_idx)

w_short = helpers.make_short_window(spec_window_type, spec_N)
w_full = helpers.place_window_on_signal_grid(w_short, total_len, start_idx)

xw_full = x_spec * w_full
x_seg = x_spec[start_idx:start_idx + spec_N]
xw_seg = x_seg * w_short

nfft_spec = 8192

Xw_spec = np.fft.fft(xw_seg, n=nfft_spec)
f_spec = np.fft.fftfreq(nfft_spec, d=1 / fs_spec)

Xw_spec_shifted = np.fft.fftshift(Xw_spec)
f_spec_shifted = np.fft.fftshift(f_spec)

mag_spec = np.abs(Xw_spec_shifted)

if show_db:
    mag_plot = 20 * np.log10(mag_spec / (np.max(mag_spec) + 1e-12) + 1e-12)
    y_title = "Magnitude (dB, normalized)"
else:
    mag_plot = mag_spec
    y_title = "Magnitude"

st.write(f"Sampling rate: {fs_spec}Hz; window length: {spec_N} samples ({spec_N/fs_spec}s)")

col1, col2 = st.columns([0.6,0.4])
with col1:
    fig_full = go.Figure()

    fig_full.add_trace(go.Scatter(
        x=t_spec,
        y=x_spec,
        mode="lines",
        name="Full sine signal",
        line=dict(color="black", width=1)
    ))

    fig_full.add_trace(go.Scatter(
        x=t_spec,
        y=w_full,
        mode="lines",
        name=f"{spec_window_type} window",
        line=dict(color="red")
    ))

    
    fig_full.update_layout(
        title="Signal",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=300
    )

    fig_full.update_layout(
    legend=dict(
        orientation="h",        # horizontal legend
        yanchor="top",
        y=-0.5,                 # move below plot (tune this)
        xanchor="center",
        x=0.5
    )
)

    st.plotly_chart(fig_full, use_container_width=True)

with col2:
    zoom_margin = 0.5
    t_start = max(0, window_start_time - zoom_margin)
    t_end = min(T_spec, window_start_time + spec_N / fs_spec + zoom_margin)

    mask_zoom = (t_spec >= t_start) & (t_spec <= t_end)

    fig_zoom = go.Figure()

    fig_zoom.add_trace(go.Scatter(
        x=t_spec[mask_zoom],
        y=x_spec[mask_zoom],
        mode="lines",
        name="Signal x[n]",
        line=dict(color="lightgray", width=1, dash="dot")
    ))

    fig_zoom.add_trace(go.Scatter(
        x=t_spec[mask_zoom],
        y=w_full[mask_zoom],
        mode="lines",
        name="Window w[n]",
        line=dict(color="red", width=2)
    ))

    fig_zoom.add_trace(go.Scatter(
        x=t_spec[mask_zoom],
        y=xw_full[mask_zoom],
        mode="lines",
        name="Windowed signal x[n]·w[n]",
        line=dict(color="black", width=2)
    ))

    fig_zoom.update_layout(
        title="Zoomed in",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=300
    )

    fig_zoom.update_layout(
        legend=dict(
            orientation="h",        # horizontal legend
            yanchor="top",
            y=-0.8,                 # move below plot (tune this)
            xanchor="center",
            x=0.5
        ))
    st.plotly_chart(fig_zoom, use_container_width=True)


fig_spec = make_subplots(
    rows=1, cols=2,
    shared_xaxes=True,
    vertical_spacing=0.12,
    subplot_titles=(
        f"Two-sided spectrum of the windowed segment ({spec_window_type}, N={spec_N})",
        "Ideal two-sided spectrum of a pure sine"
    )
)

fig_spec.add_trace(
    go.Scatter(
        x=f_spec_shifted,
        y=mag_plot,
        mode="lines",
        name=f"{spec_window_type} windowed spectrum",
        line=dict(color="red"),
    ),
    row=1, col=1
)

ymin = float(np.min(mag_plot))
ymax = float(np.max(mag_plot))

fig_spec.add_trace(
    go.Scatter(
        x=[f0,f0],
        y=[ymin, ymax],
        mode="lines",
        line=dict(dash="dot", color="black"),
        name=f"+{f0} Hz"
    ),
    row=1, col=1
)

fig_spec.add_trace(
    go.Scatter(
        x=[-f0, -f0],
        y=[ymin, ymax],
        mode="lines",
        line=dict(dash="dot", color="black"),
        name=f"-{f0} Hz"
    ),
    row=1, col=1
)

ideal_freqs = np.array([-f0, f0])
ideal_amps = np.array([max(mag_plot),max(mag_plot)])

for f0_i, a0_i in zip(ideal_freqs, ideal_amps):
    # stem line
    fig_spec.add_trace(
        go.Scatter(
            x=[f0_i, f0_i],
            y=[0, a0_i],
            mode="lines",
            line=dict(color="red"),
            showlegend=False
        ),
        row=1, col=2
    )

    fig_spec.add_trace(
        go.Scatter(
            x=[f0_i],
            y=[a0_i],
            mode="markers",
            marker=dict(color="black", size=10),
            showlegend=False
        ),
        row=1, col=2
    )

fig_spec.update_layout(
    xaxis_title="Frequency (Hz)",
    template="plotly_white",
    height=350,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.8,
        xanchor="center",
        x=0.5
    ),
    margin=dict(b=90)
)

fig_spec.update_xaxes(range=[-5, 5], row=1, col=1)
fig_spec.update_xaxes(range=[-5, 5], title="Frequency (Hz)", row=1, col=2)

fig_spec.update_yaxes(title=y_title, row=1, col=1)
fig_spec.update_yaxes(title="Ideal amplitude", row=1, col=2)

if show_db:
    fig_spec.update_yaxes(range=[-80, 5], row=1, col=1)

st.plotly_chart(fig_spec, use_container_width=True)



st.header("5. Choosing a window")


col1, col2, col3, col4 = st.columns(4)

with col1:
    scenario = st.selectbox(
        "Scenario",
        [
            "Noisy single tone",
            "Strong distant interferer",
            "Strong nearby interferer",
            "Two close equal tones",
            "Two close unequal tones",
            "Broadband / flat spectrum",
        ],
        key="single_window_scenario"
    )

with col2:
    chosen_window = st.selectbox(
        "Window type",
        ["Rectangular", "Hann", "Hamming", "Blackman"],
        key="single_window_type"
    )

with col3:
    N_demo = st.slider(
        "Window length N",
        min_value=32,
        max_value=512,
        value=128,
        step=2,
        key="single_window_length"
    )

with col4:
    show_db_demo = st.checkbox(
        "Show spectrum in dB",
        value=True,
        key="single_window_show_db"
    )

col5, col6 = st.columns(2)

with col5:
    noise_std = st.slider(
        "Noise strength",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        key="single_window_noise"
    )

with col6:
    delta_f = st.slider(
        "Frequency spacing / nearby offset (Hz)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="single_window_deltaf"
    )

fs_demo = 200
T_demo = 4.0
t_demo = np.arange(0, T_demo, 1 / fs_demo)
rng = np.random.default_rng(0)

x_demo, scenario_text = helpers.generate_scenario_signal(
    scenario=scenario,
    t=t_demo,
    fs=fs_demo,
    rng=rng,
    noise_std=noise_std,
    delta_f=delta_f
)

start_idx_demo = len(x_demo) // 2 - N_demo // 2
start_idx_demo = max(0, min(start_idx_demo, len(x_demo) - N_demo))
end_idx_demo = start_idx_demo + N_demo

t_start_demo = start_idx_demo / fs_demo
t_end_demo = end_idx_demo / fs_demo

seg_demo = x_demo[start_idx_demo:end_idx_demo]
w_demo = helpers.make_analysis_window(chosen_window, N_demo)
xw_demo = seg_demo * w_demo

w_full_demo = helpers.place_window(w_demo, len(x_demo), start_idx_demo)
xw_full_demo = np.zeros_like(x_demo)
xw_full_demo[start_idx_demo:end_idx_demo] = xw_demo

nfft_demo = 16384
X_demo = np.fft.fft(xw_demo, n=nfft_demo)
f_demo = np.fft.fftfreq(nfft_demo, d=1 / fs_demo)

X_demo_shift = np.fft.fftshift(X_demo)
f_demo_shift = np.fft.fftshift(f_demo)

mag_demo = np.abs(X_demo_shift)

if show_db_demo:
    y_demo = 20 * np.log10(mag_demo / (np.max(mag_demo) + 1e-12) + 1e-12)
    y_title_demo = "Magnitude (dB, normalized)"
else:
    y_demo = mag_demo
    y_title_demo = "Magnitude"

st.markdown(f"**Scenario:** {scenario_text}")

left_col, right_col = st.columns(2)

with left_col:
    fig_time_demo = go.Figure()

    fig_time_demo.add_trace(go.Scatter(
        x=t_demo,
        y=x_demo,
        mode="lines",
        name="Signal",
        line=dict(color="lightgray")
    ))

    fig_time_demo.add_trace(go.Scatter(
        x=t_demo,
        y=w_full_demo,
        mode="lines",
        name=f"{chosen_window} window",
        line=dict(color="red", width=3)
    ))

    fig_time_demo.add_trace(go.Scatter(
        x=t_demo,
        y=xw_full_demo,
        mode="lines",
        name="Windowed signal",
        line=dict(color="black", width=2)
    ))

    fig_time_demo.add_vrect(
        x0=t_start_demo,
        x1=t_end_demo,
        fillcolor="red",
        opacity=0.08,
        line_width=0,
    )

    fig_time_demo.update_layout(
        title=f"Signal and selected analysis window (N = {N_demo} samples = {N_demo/fs_demo:.2f} s)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=90)
    )

    st.plotly_chart(fig_time_demo, use_container_width=True)

with right_col:
    fig_spec_demo = go.Figure()

    fig_spec_demo.add_trace(go.Scatter(
        x=f_demo_shift,
        y=y_demo,
        mode="lines",
        name=f"{chosen_window} windowed spectrum",
        line=dict(color="red")
    ))

    reference_freqs = []
    if scenario in ["Noisy single tone", "Strong distant interferer", "Strong nearby interferer",
                    "Two close equal tones", "Two close unequal tones"]:
        reference_freqs.append(20)
    if scenario == "Strong distant interferer":
        reference_freqs.append(60)
    if scenario in ["Strong nearby interferer", "Two close equal tones", "Two close unequal tones"]:
        reference_freqs.append(20 + delta_f)

    for f_ref in reference_freqs:
        fig_spec_demo.add_trace(go.Scatter(
            x=[f_ref, f_ref],
            y=[np.min(y_demo), np.max(y_demo)],
            mode="lines",
            line=dict(color="black", dash="dot"),
            name=f"+{f_ref:.1f} Hz"
        ))
        fig_spec_demo.add_trace(go.Scatter(
            x=[-f_ref, -f_ref],
            y=[np.min(y_demo), np.max(y_demo)],
            mode="lines",
            line=dict(color="black", dash="dot"),
            name=f"-{f_ref:.1f} Hz"
        ))

    fig_spec_demo.update_layout(
        title="Two-sided spectrum of the windowed segment",
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_title_demo,
        template="plotly_white",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=90)
    )

    fig_spec_demo.update_xaxes(range=[-40, 40])

    if show_db_demo:
        fig_spec_demo.update_yaxes(range=[-120, 5])

    st.plotly_chart(fig_spec_demo, use_container_width=True)




st.header("6. Amplitude stability versus frequency alignment")


col1, col2, col3 = st.columns(3)

with col1:
    fs_demo = st.slider("Sampling rate (Hz)", 100, 500, 200, key="cmp_fs")

with col2:
    N_demo = st.slider("Window length N", 32, 512, 128, step=2, key="cmp_N")

with col3:
    freq_offset = st.slider("Frequency offset (Hz)", 0.0, 2.0, 0.3, step=0.05, key="cmp_offset")


f_bin = 20.0
f0 = f_bin + freq_offset

t = np.arange(N_demo) / fs_demo
x = np.sin(2 * np.pi * f0 * t)

window_types = ["Rectangular", "Hann", "Hamming", "Blackman"]

def make_window(wtype, N):
    if wtype == "Rectangular":
        return np.ones(N)
    elif wtype == "Hann":
        return np.hanning(N)
    elif wtype == "Hamming":
        return np.hamming(N)
    elif wtype == "Blackman":
        return np.blackman(N)
    else:
        raise ValueError(f"Unknown window type: {wtype}")


nfft = 8192
spectra = {}
peak_values = {}

for wname in window_types:
    w = make_window(wname, N_demo)
    xw = x * w
    xw = xw / np.sqrt(np.sum(w**2))

    X = np.fft.fft(xw, n=nfft)
    f = np.fft.fftfreq(nfft, d=1 / fs_demo)

    Xs = np.fft.fftshift(X)
    fs_shift = np.fft.fftshift(f)

    mag = np.abs(Xs)
    spectra[wname] = (fs_shift, mag)
    peak_values[wname] = np.max(mag)

left_col, right_col = st.columns(2)
colors = ["black", "darkred", "red", "orange"]
styles = ["solid", "dash", "dashdot", "longdash"]
with left_col:
    fig_spec = go.Figure()

    for w, wname in enumerate(window_types):
        f_plot, mag_plot = spectra[wname]
        fig_spec.add_trace(go.Scatter(
            x=f_plot,
            y=mag_plot,
            mode="lines",
            line=dict(color=colors[w], width=1.5, dash=styles[w]),
            name=wname
        ))

    fig_spec.add_trace(go.Scatter(
        x=[f0, f0],
        y=[0, max(peak_values.values()) * 1.05],
        mode="lines",
        line=dict(color="black", dash="dot"),
        name=f"+{f0:.2f} Hz"
    ))

    fig_spec.add_trace(go.Scatter(
        x=[-f0, -f0],
        y=[0, max(peak_values.values()) * 1.05],
        mode="lines",
        line=dict(color="black", dash="dot"),
        name=f"-{f0:.2f} Hz"
    ))

    fig_spec.update_layout(
        title=f"Spectra (f = {f0:.2f} Hz)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        template="plotly_white",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.45,
            xanchor="center",
            x=0.5
        ),
        margin=dict(b=100)
    )

    fig_spec.update_xaxes(range=[f_bin - 10, f_bin + 10])

    st.plotly_chart(fig_spec, use_container_width=True)

with right_col:
    fig_bar = go.Figure()

    bar_x = list(peak_values.keys())
    bar_y = list(peak_values.values())

    fig_bar.add_trace(go.Bar(
        x=bar_x,
        y=bar_y,
        text=[f"{v:.2f}" for v in bar_y],
        textposition="outside"
    ))

    fig_bar.update_layout(
        title="Peak amplitude for each window",
        xaxis_title="Window",
        yaxis_title="Peak magnitude",
        template="plotly_white",
        height=350,
        margin=dict(t=60, b=40)
    )
    fig_bar.update_traces(marker_color='rgb(255,50,0)', marker_line_color='rgb(200,25,0)',
                  marker_line_width=1.5, opacity=0.6)

    fig_bar.update_yaxes(range=[0, max(bar_y) * 1.2])

    st.plotly_chart(fig_bar, use_container_width=True)