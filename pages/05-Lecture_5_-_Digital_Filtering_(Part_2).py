import streamlit as st
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import src.helpers as helpers


st.set_page_config(layout="wide")

st.title("Interactive Filter Design: Laplace, Fourier, and Z-Transform")

with st.expander("The Laplace Transform"):
    
    st.header("The Laplace Transform")
    st.markdown(r"""
        The **Laplace Transform** can be understood (in part) as a **generalization** of 
        the Fourier Transform. For any continuous (!) function $g(t)$, it takes the shape:

        $\mathcal{L}\{g(t)\}=F(s)=\int_0^\infty g(t) exp(-st)dt$

        and corresponds to a function of the complex variable $s\in\mathbb{C}$.

        It's shape is quite similar to the Fourier Transform ("Continuous Time Fourier 
        Transform") of a function $g(t)$, which is defined as:

        $\mathcal{F}\{g(t)\} = G(f) = \int_{-\infty}^{\infty} g(t) exp(-2\pi i ft) dt$

        with the real-world interpretation of $t$ describing time, $f$ describing frequency.

        As you can see, the range of the integral differes - but also the exponential function
        $exp()$ inside the integral. For the Fourier Transform, this exponential is purely 
        *complex-valued* - it exclusively describes complex sinusoids (i.e., *oscillation*).
        For the DFT, we used these complex sinusoids as 'test functions' of pure frequencies with
        which we tested whether the sinusoid's frequency was present in an input signal $x(t)$
        or not.

        **Importantly**, the use of the *purely complex* exponential in the Fourier 
        Transform means that we try to estimate the similarity between our test signal and
        pure sinusoids of amplitude 1 that *to not decay*.  
        In contrast, we *do* allow decay or growth of our exponential in the Laplace 
        Transform, making it a *generalization* of sorts of the Fourier Transform. We 
        suddenly not only check whether a sine/cosine linear combination is present in our 
        signal, but also whether it *grows* or *decays* over time!

    """)

    st.header("Understanding the exponentials")

    st.markdown(r"""
        We'll first discuss the expression $exp(-st), s=\sigma_1 + i\sigma_2\in \mathbb{C}$, that is part of the
        Laplace Transform.
    """)

    col1, col2 = st.columns([0.6,0.4])

    with col1:
        sigma1 = st.number_input(r"Real part $\sigma_1$", min_value=-1.0, max_value=1.0, step=0.05, value=0.0)
        sigma2 = st.number_input(r"Imaginary part $\sigma_2= 2\pi f$", min_value=-30.0, max_value=30.0, step=0.05, value=6.28)
        t_max = st.number_input("Duration (s)", min_value=1.0, max_value=100.0)
        dt = 0.01

        st.markdown(r"""
            **A few comments:**
            - The expression $exp(-st)$ can be split into a real and a complex exponential 
            part: $exp(-st)=exp(-(\sigma_1 + i\sigma_2)t)=exp(-\sigma_1 t)\cdot exp(-i\sigma_2 t)$
            - The real part of $s$ tells us whether the exponential is *decaying* or 
            *growing*:  
            - if $\sigma_1>0$, then $exp(-\sigma_1 t)$ decreases over time (decay)
            - if $\sigma_1<0$, then $exp(-\sigma_1 t)$ increases over time (growth)
            - The imaginary part of $s$ tells us which frequency the complex exponential
            is oscillating with.

            Together, both contributions make $exp(-st)$ *spiral* in the s-plane!
        """)
    s = sigma1 + 1j * sigma2
    t = np.arange(0, t_max, dt)
    t = np.linspace(0, t_max, 1000)
    color_values = list(range(1000))
    exp_vals = np.exp(-s * t)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.real(exp_vals),
            y=np.imag(exp_vals),
            opacity=0.8,
            marker=dict(
                size=10,
                cmax=1000,
                cmin=0,
                color=color_values,
                colorbar=dict(
                    title=dict(text="Time point")
                ),
                colorscale="Cividis"
            ),
            mode="markers+lines",
            name="exp(-st)",
        ))

        fig.update_layout(
            title="s-plane",
            width=600,
            height=600,                 
            xaxis=dict(
                title="Real axis",
                range=[-1.5, 1.5],
                scaleanchor="y",        
                scaleratio=1
            ),
            yaxis=dict(
                title="Imaginary axis",
                range=[-1.5, 1.5],
            )
        )
        st.plotly_chart(fig, width="content")


    st.header("Understanding the Laplace transform")

    st.markdown(r"""
    We'll take a look at what the Laplace transform does to an exponential signal:

    $f(t)=\exp\big((\sigma_1+i\sigma_2)t\big)=\exp(\sigma_1 t)\exp(i\sigma_2 t)$,

    (i.e., an oscillation whose magnitude may grow or decay over time).

    Its Laplace transform is:

    $
    \mathcal{L}\{f(t)\}=F(s) = \int_0^\infty f(t) exp(-st) dt
    = \int_0^\infty \exp((\sigma_1+i\sigma_2)t)\exp(-st)\,dt
    = \int_0^\infty \exp\big(((\sigma_1+i\sigma_2)-s)t\big)\,dt
    = \frac{1}{s-(\sigma_1+i\sigma_2)}.
    $

    So the transform has a **pole** at: $s=\sigma_1+i\sigma_2$.


    This means the pole location directly reflects the signal's:
    - **real part**: growth or decay
    - **imaginary part**: oscillation frequency
    """)

    sigma1 = st.slider(
        r"Choose $\sigma_1$ (decay/amplification)",
        min_value=-10.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    sigma2 = st.slider(
        r"Choose $\sigma_2$ (oscillation)",
        min_value=-6.0,
        max_value=6.0,
        value=6.0,
        step=0.1
    )

    sigma_vals = np.linspace(-10, 10, 120)
    omega_vals = np.linspace(-10, 10, 120)
    SIGMA, OMEGA = np.meshgrid(sigma_vals, omega_vals)

    S = SIGMA + 1j * OMEGA
    G = 1 / (S - (sigma1 + 1j * sigma2))

    # Magnitude
    G_mag = np.abs(G)

    # Clip to keep the pole visible without dominating everything
    zmax = 8.0
    G_mag = np.clip(G_mag, 0, zmax)

    col1, col2 = st.columns(2)

    with col1:
        fig3d = go.Figure()

        fig3d.add_trace(go.Surface(
            x=SIGMA,
            y=OMEGA,
            z=G_mag,
            opacity=0.65,
            colorscale="Jet",
            showscale=True,
            colorbar=dict(title="|F(s)|"),
            
        ))

        # Mark the pole location
        fig3d.add_trace(go.Scatter3d(
            x=[sigma1, sigma1],
            y=[sigma2, sigma2],
            z=[0, zmax],
            mode="lines+markers",
            line=dict(color="red", width=8),
            marker=dict(size=4, color="red"),
            name="Pole at s = jω₀"
        ))

        fig3d.update_layout(
            title=dict(
                text=r"Surface plot"
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text=r"Real"),
                    backgroundcolor="rgb(240,240,240)",
                ),
                yaxis=dict(
                    title=dict(text=r"Imag"),
                    backgroundcolor="rgb(240,240,240)",
                ),
                zaxis=dict(
                    title=dict(text=r"|F(s)|"),
                    backgroundcolor="rgb(245,245,245)",
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
                aspectmode="cube",
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        st.plotly_chart(fig3d, use_container_width=True)


    with col2:
        fig2d = go.Figure()

        fig2d.add_trace(go.Contour(
            x=sigma_vals,
            y=omega_vals,
            z=G_mag,
            colorscale="turbo",
            opacity=0.9,
            zmin=0,
            zmax=zmax,
            contours=dict(
                start=0,
                end=zmax,
                size=0.5,
                showlabels=True,
                labelfont=dict(size=10, color="white")
            ),
            colorbar=dict(title="|F(s)|")
        ))

        # Pole marker
        fig2d.add_trace(go.Scatter(
            x=[sigma1],
            y=[sigma2],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name="Pole at s = iω"
        ))

        fig2d.update_layout(
            title="Contour plot",
            xaxis=dict(
                title="Real",
                constrain="domain",
                scaleanchor="y",
            ),
            yaxis=dict(
                title="Imag",
                scaleanchor=None,
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        st.plotly_chart(fig2d, use_container_width=True)


with st.expander("From Laplace to the Z-transform"):
    st.header("From Laplace to the Z-transform")

    st.markdown(r"""
    So far, we worked with continuous-time signals $x(t)$ and the Laplace transform:
    $X(s)=\int_0^\infty x(t)e^{-st}dt$. This compares a signal to 
    **continuous exponentials** $exp(st)$.

    But in digital signal processing, signals $x[n]$ are **discrete**; so instead of 
    exponentials $exp(st)$ in continuous time $t\in\mathbb{R}$, we use their discrete counterpart
    in time samples $n\in\mathbb{Z}$:

    $exp(sn) = exp((\sigma_1 + i\sigma_2)n) = exp(\sigma_1 + i\sigma_2)^n = z^n, \quad z \in \mathbb{C}.
    $

    This leads directly to the **Z-transform**, which returns a *continuous* function of 
    the complex input $z$ for a given *discrete* signal $x[n]$:

    $
    X(z)=\sum_{n=-\infty}^{\infty} x[n]z^{-n}.
    $

    ---

    As $z\in\mathbb{C}$, we can write it as $z = r e^{i\omega}$, with an amplitude $r$
    and a phase $\omega$. So just like in the Laplace transform:
    - $r$ controls **growth or decay**
    - $\omega$ controls the **oscillation**

    The Z-transform therefore measures how much a discrete signal $x[n]$ behaves like a
    *discrete exponential* that may grow, decay, or oscillate.
    """)


with st.expander("From Z-transform to the DFT"):
    st.header("From Z-transform to the DFT")

    st.markdown(r"""
        The Z-transform evaluates a signal for any complex value $z$:  

        $
        X(z)=\sum_{n=-\infty}^{\infty} x[n] z^{-n} = \sum_{n=-\infty}^{\infty} x[n] r^n e^{i\omega n}.
        $

        If we are only interested in **frequencies**, not in whether the signal grows
        or decays, we can just set $r = 1 \Rightarrow z = e^{i\omega}.
        $

        This already looks quite familiar, though:

        $X(\omega) = \sum_{n=0}^{N-1} x[n] e^{-i \omega n}.$

        What we're seeing here is actually the **Discrete-time Fourier Transform (DTFT)** - 
        the Fourier Transform where we go from discrete time signal to *continuous*
        frequency signal. If we sample the frequencies as well, we're directly at the 
        DFT we know so well by now:

        $X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi kn/N}.$

        ---

        ### To sum up:

        - The Z-transform tells us how similar our discrete input signal $x[n]$ is to:
          - growing ($r>1$)
          - unaltered ($r=1$)
          - decaying ($r<1$)  

          sinusoids $r exp(i\omega)$.
          The Z-plane gives us the (discrete) **full story** whereever we see poles, there
          is for sure a similarity to signals of this shape.  
          (The Laplace Transform would give us the full *continuous* story.)
        - The DTFT corresponds to the Z-transform where we only evaluate similarity to:
          - unaltered ($r=1$), i.e., *pure*  

          sinusoids $exp(i\omega)$. These signals don't grow or decay - they retain their
          sinusoidal shape forever. The DTFT thus just tells us whether a sinusoid of a
          specific frequency is part of our signal $x[n]$, but it doesn't tell us about
          whether this frequency component dies down or grows in our signal or not.  
          (The Continuous-time Fourier Transform (CTFT) would give us the *continuous* version
          of similarity to 'static' sines and cosines here.)

        - The DFT corresponds to a frequency-sampled version of the DTFT then.
        """)



st.header("Filtering signals")

st.markdown(r"""
    One method of filtering signals is called **frequency sampling**. Here, we sample 
    $n_fft$ points between $-f_s/2$ and $+f_s/2$ of an ideal lowpass filter's
    frequency response for a given cutoff frequency $f_c$, then we take the inverse
    DFT of this (the *impulse response* $h[n]=iDFT(H[k])$) and linearly convolve
    our signal with it. Importantly, the iDFT of a rectangular function is a 
    **cardinal sine**, $h[n] = sin(2\pi f_c n)$, which oscillates at the cutoff 
    frequency $f_c$ and introduces oscillations
    in our signal during the convolution.
    
    **Pre-ringing**  

    If $h[n]$ is centered around sample 0, the filter is **acausal**, meaning the
    impulse response uses past *and* future values for convolution. As a result,
    we hear **pre-ringing** (i.e., wriggles before the cause even starts up) 
    before sharp transitions in time domain (like a drum kick). 
    - The longer the filter (i.e., its impulse response), the more samples of our
    signal are affected at once by the sinc's oscillation, and the sooner we will
    hear and see preringing.
    - As the sinc oscillates with $f_s$, this will be nicely *audible* if the cutoff
    frequency ranges in, e.g., the vocal range (between ~100Hz and 1000Hz).

    To get rid of this, we can just shift the impulse response to the right (such that
    the full windowed response is located at samples > 0). In this case, we won't hear
    pre-ringing anymore; the filter is **causal** then.

    Some audio suggestions to check out; just paste the url below:
    - [Default: Some exemplary drum kicks](https://www.youtube.com/watch?v=thapspq3qVQ)
    - [Pure 440Hz tone for reference (yes, this is really what we hear as ringing for
    a cutoff of 440Hz - pretty cool, huh)](https://www.youtube.com/watch?v=OUvlamJN3nM)
    - [Disturbed - Down with the sickness](https://www.youtube.com/watch?v=09LTT0xwdfw&list=RD09LTT0xwdfw&start_radio=1)
""")


youtube_url = st.text_input("Enter YouTube URL", value="https://www.youtube.com/watch?v=thapspq3qVQ")




if youtube_url:
    col1, col2 = st.columns(2)
    with col1:
        fc = st.number_input("Cutoff frequency (Hz)",value=440, min_value=100,
        max_value=1000)
    with col2:
        N = int(st.number_input("Filter length (#samples)", value=4001))
    title, result = helpers.download_song(youtube_url)
    x, sr = sf.read(result)
    if x.ndim > 1:  # convert to mono if stereo
        x = x.mean(axis=1) 
    if len(x) > sr * 30:    # truncate if longer than 30s
        x = x[:sr * 30]
    n_fft = 40000*2+1
    h, H, f = helpers.ideal_lowpass_ir(fc=fc, sr=sr, n_fft=n_fft)
    center = n_fft // 2
    start = center - N // 2
    stop = start + N
    h = h[start:stop]
    h = h / np.sum(h)

    with st.expander("Ringing of a filter"):
        st.subheader("Impulse response centered around sample 0: Ringing")


        col1, col2, col3 = st.columns(3)
        with col1:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=np.arange(len(h)) - len(h)//2,   # centered around 0
                y=h,
                mode="lines",
                name="h[n]",
                line=dict(color="black")
            ))

            fig_h.update_layout(
                title="Impulse response of an ideal Lowpass Filter",
                xaxis_title="Sample index n",
                yaxis_title="Amplitude",
            )
            st.plotly_chart(fig_h, width='content')

        with col2:
            fig_H = go.Figure()
            fig_H.add_trace(go.Scatter(
                x=f,
                y=np.abs(H),
                line=dict(color="black"),
                name="|H[k]|"
            ))

            fig_H.update_layout(
                title="Frequency response of an ideal Lowpass Filter",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                xaxis=dict(range=[-1500,1500])
            )
            st.plotly_chart(fig_H, width='content')

        with col3:
            fig_H = go.Figure()
            fig_H.add_trace(go.Scatter(
                x=f,
                y=np.angle(H),
                line=dict(color="black"),
                name="∠H[k]"
            ))

            fig_H.update_layout(
                title="Phase response of an ideal Lowpass Filter",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Phase (unwrapped; radians)",
                xaxis=dict(range=[-1500,1500])
            )
            st.plotly_chart(fig_H, width='content')


        y_full = np.convolve(x, h, mode="full")
        M = len(h) // 2
        y = y_full[M:M + len(x)]

        y = y / (np.max(np.abs(y)) + 1e-9)



        t_x = np.arange(len(x)) / sr
        t_y = np.arange(len(y)) / sr

        figlp = go.Figure()
        figlp.add_trace(go.Scatter(
            x=t_x,
            y=x,
            mode="lines",
            name="x (original)",
            line=dict(color="grey")
        ))

        figlp.add_trace(go.Scatter(
            x=t_y,
            y=y,
            mode="lines",
            name="y (filtered)",
            line=dict(color="red")
        ))

        figlp.update_layout(
            title="Filtered signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            xaxis=dict(range=[0.98,1.16])
        )

        col1, col2 = st.columns([0.6,0.4], vertical_alignment="center")
        with col1:
            st.plotly_chart(figlp, width='content')

        with col2: 
            st.markdown("**Original track:**")
            st.audio(x, sample_rate=sr)

            st.markdown("**Filtered track:**")
            st.audio(y, sample_rate=sr)



        st.markdown(r"""
            We clearly see the oscillations starting up *before* the drum kick even happens!  
            $\rightarrow$ That's **pre-ringing**!  
        """)

    with st.expander("Fixing the ringing"):
        st.subheader("a) Truncating the impulse response")
        st.markdown(r"""
            To remove that effect, we could 'cut off' the negative part of the impulse
            response. This works nicely if you look at the signal and its filtered version; 
            the issue, though? This is now a *novel* filter - the frequency response looks
            *entirely different* than what we started out with.
        """)



        h_causal = np.zeros([2*len(h),])
        h_causal[:N//2+1] = h[N//2:]
        H_causal = np.fft.fftshift(np.fft.fft(h_causal, n=len(H)))
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_h2 = go.Figure()
            fig_h2.add_trace(go.Scatter(
                x=np.arange(len(h_causal)),  
                y=h_causal,
                mode="lines",
                name="h[n]",
                line=dict(color="black")
            ))

            fig_h2.update_layout(
                title="Altered impulse response (starting at sample 0 now)",
                xaxis_title="Sample index n",
                yaxis_title="Amplitude",
            )
            st.plotly_chart(fig_h2, width='content', key="yolo")
        with col2:
            fig_H2 = go.Figure()
            fig_H2.add_trace(go.Scatter(
                x=f,
                y=np.abs(H_causal),
                name="H[k] causal",
                line=dict(color="black")
            ))

            fig_H2.update_layout(
                title="Frequency response to the altered impulse response",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                xaxis=dict(range=[-1500,1500])
            )

            st.plotly_chart(fig_H2, width='content', key="yo")
        with col3:
            fig_H = go.Figure()
            fig_H.add_trace(go.Scatter(
                x=f,
                y=np.unwrap(np.angle(H_causal)),
                line=dict(color="black"),
                name="∠H[k]"
            ))

            fig_H.update_layout(
                title="Phase response of an ideal Lowpass Filter",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Phase (unwrapped; radians)",
                xaxis=dict(range=[-1500,1500])
            )
            st.plotly_chart(fig_H, width='content')

        y = np.convolve(x, h_causal, mode="full")
        y = y / (np.max(np.abs(y)) + 1e-9)

        t_x = np.arange(len(x)) / sr
        t_y = np.arange(len(y)) / sr

        figlp2 = go.Figure()
        figlp2.add_trace(go.Scatter(
            x=t_x,
            y=x,
            mode="lines",
            name="x (original)",
            line=dict(color="grey")
        ))

        figlp2.add_trace(go.Scatter(
            x=t_y,
            y=y,
            mode="lines",
            name="y (filtered)",
            line=dict(color="red")
        ))

        figlp2.update_layout(
            title="Filtered signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            xaxis=dict(range=[0.98,1.16])
        )

        col1, col2 = st.columns([0.6,0.4], vertical_alignment="center")
        with col1:
            st.plotly_chart(figlp2, width='content', key="swag")

        with col2: 
            st.markdown("**Original track:**")
            st.audio(x, sample_rate=sr)

            st.markdown("**Filtered track:**")
            st.audio(y, sample_rate=sr)



        st.markdown(r"""
            We fixed the preringing now, but the frequency response to our adapted impulse
            response looks nothing like the ideal lowpass we'd like to have anymore: there's
            a slump in the passband we don't actually want, and above all, the stopband
            is not attenuated anymore: high frequencies can thus 'slip through' the filter.
        """)


        st.subheader("b) Windowing the impulse response")

        window_type = st.selectbox(
            "Choose window type",
            ["Rectangular", "Hann", "Hamming", "Blackman"]
        )
        window_length = int(st.number_input("Window length", value=N//2))

        w = helpers.make_short_window(window_type, window_length)

        window = np.zeros(N)
        start = (N - len(w)) // 2
        end = start + len(w)
        window[start:end] = w
        h_windowed = h * window

        Hlen = len(H)

        h_pad = np.zeros(len(H))
        start = (len(H) - len(h_windowed)) // 2
        h_pad[start:start + len(h_windowed)] = h_windowed
        H_windowed = np.fft.fftshift(np.fft.fft(h_pad))

        phi0 = np.angle(H_windowed)
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_h2 = go.Figure()

            fig_h2.add_trace(go.Scatter(
                x=np.arange(len(h)) - len(h)//2,
                y=h,
                mode="lines",
                name="h[n]",
                line=dict(color="grey")
            ))

            fig_h2.add_trace(go.Scatter(
                x=np.arange(len(h_windowed)) - len(h_windowed)//2,
                y=h_windowed,
                mode="lines",
                name="h[n]*w[n]",
                line=dict(color="red")
            ))

            fig_h2.update_layout(
                title="Impulse response",
                xaxis_title="Sample index n",
                yaxis_title="Amplitude",
            )
            st.plotly_chart(fig_h2, width='content', key="yolo2")
        with col2:
            fig_H2 = go.Figure()
            fig_H2.add_trace(go.Scatter(
                x=f,
                y=np.abs(H),
                name="|H[k]|",
                line=dict(color="black")
            ))

            fig_H2.add_trace(go.Scatter(
                x=f,
                y=np.abs(H_windowed),
                name="|H[k]| windowed",
                line=dict(color="red")
            ))

            fig_H2.update_layout(
                title="Frequency response",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Phase (unwrapped; radians)",
                xaxis=dict(range=[-1500,1500])
            )

            st.plotly_chart(fig_H2, width='content', key="yo2")
        with col3:
            fig_H2 = go.Figure()
            fig_H2.add_trace(go.Scatter(
                x=f,
                y=np.unwrap(np.angle(H)),
                name="∠H[k]",
                line=dict(color="black", dash="dot")
            ))
            fig_H2.add_trace(go.Scatter(
                x=f,
                y=np.unwrap(np.angle(H)),
                name="∠H[k] windowed",
                line=dict(color="red", dash="solid")
            ))

            fig_H2.update_layout(
                title="Phase response of an ideal Lowpass Filter",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                xaxis=dict(range=[-1500,1500])
            )
            st.plotly_chart(fig_H2, width='content')


        y_wn = np.convolve(x, h_windowed, mode="full")
        M = len(h_windowed) // 2
        y_wn = y_wn[M:M + len(x)]
        y_wn = y_wn / (np.max(np.abs(y_wn)) + 1e-9)

        y = np.convolve(x, h_windowed, mode="full")
        M = len(h_windowed) // 2
        y = y_full[M:M + len(x)]
        y = y / (np.max(np.abs(y)) + 1e-9)


        t_x = np.arange(len(x)) / sr
        t_y = np.arange(len(y)) / sr

        figlp2 = go.Figure()
        figlp2.add_trace(go.Scatter(
            x=t_x,
            y=x,
            mode="lines",
            name="x (original)",
            line=dict(color="grey")
        ))

        figlp2.add_trace(go.Scatter(
            x=t_y,
            y=y,
            mode="lines",
            name="y (filtered)",
            line=dict(color="red")
        ))

        figlp2.add_trace(go.Scatter(
            x=t_y,
            y=y_wn,
            mode="lines",
            name="y (filtered with window)",
            line=dict(color="orange")
        ))

        figlp2.update_layout(
            title="Filtered signal",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            xaxis=dict(range=[0.98,1.16])
        )

        col1, col2 = st.columns([0.6,0.4], vertical_alignment="center")
        with col1:
            st.plotly_chart(figlp2, width='content', key="swag2")

        with col2: 
            st.markdown("**Original track:**")
            st.audio(x, sample_rate=sr)

            st.markdown("**Filtered track:**")
            st.audio(y_wn, sample_rate=sr)



        st.markdown(r"""
            Using a window, we can alleviate the ripples in time domain that are
            introduced by harsh edges in the spectrum. The resulting frequency response
            is a more lenient lowpass filter, but 
        """)


st.header("Phase and time delay")


fs = 1000
t = np.linspace(0, 1, fs)
f0 = 5  # Hz
phi = np.pi / 2  # phase shift

x = np.sin(2 * np.pi * f0 * t)
x_shifted = np.sin(2 * np.pi * f0 * t + phi)

tau = phi / (2 * np.pi * f0)
x_delayed = np.sin(2 * np.pi * f0 * (t + tau))


fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=x, name="Original signal", line=dict(color="grey")))
fig.add_trace(go.Scatter(
    x=t,
    y=x_shifted,
    name="Phase shifted (φ = π/2)",
    line=dict(dash="solid", color="red")
))
fig.add_trace(go.Scatter(
    x=t,
    y=x_delayed,
    name="Time shifted (τ)",
    line=dict(dash="dash", color="darkred")
))
fig.update_layout(
    title="Phase shift",
    xaxis_title="Time (s)",
    yaxis_title="Amplitude"
)

st.plotly_chart(fig, width='content', key="swag3")