# -*- coding: utf-8 -*-
"""Dynamic process models for microscopy simulation."""
import numpy as np


# ---------------------------------------------------------------------------
# Particle motion
# ---------------------------------------------------------------------------

def brownian_motion(n_particles, n_frames, D, dt=0.03, dim=3,
                    bounds=None):
    """Simulate Brownian motion.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_frames : int
        Number of time frames.
    D : float
        Diffusion coefficient (µm²/s).
    dt : float
        Frame interval in seconds.
    dim : int
        Number of spatial dimensions (2 or 3).
    bounds : tuple or None
        ((min0, max0), (min1, max1), ...) reflecting boundaries.

    Returns
    -------
    ndarray
        Positions array of shape (n_particles, n_frames, dim).
    """
    sigma = np.sqrt(2 * D * dt)
    steps = np.random.randn(n_particles, n_frames, dim) * sigma
    positions = np.cumsum(steps, axis=1)

    if bounds is not None:
        for d_idx in range(min(dim, len(bounds))):
            lo, hi = bounds[d_idx]
            # Reflecting boundaries
            for _ in range(10):  # iterative reflection
                below = positions[:, :, d_idx] < lo
                above = positions[:, :, d_idx] > hi
                positions[:, :, d_idx][below] = 2 * lo - positions[:, :, d_idx][below]
                positions[:, :, d_idx][above] = 2 * hi - positions[:, :, d_idx][above]
                if not (np.any(below) or np.any(above)):
                    break

    return positions


def directed_motion(n_particles, n_frames, D, velocity, dt=0.03,
                    dim=3):
    """Simulate directed motion with diffusion.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_frames : int
        Number of frames.
    D : float
        Diffusion coefficient (µm²/s).
    velocity : array-like
        Drift velocity vector (µm/s), shape (dim,).
    dt : float
        Frame interval in seconds.
    dim : int
        Spatial dimensions.

    Returns
    -------
    ndarray
        Positions (n_particles, n_frames, dim).
    """
    velocity = np.asarray(velocity)[:dim]
    # Brownian component
    positions = brownian_motion(n_particles, n_frames, D, dt, dim)
    # Add drift
    drift = np.outer(np.arange(n_frames), velocity * dt)  # (T, dim)
    positions += drift[np.newaxis, :, :]
    return positions


def confined_motion(n_particles, n_frames, D, confinement_radius,
                    dt=0.03, dim=3):
    """Simulate confined Brownian motion.

    Uses Ornstein-Uhlenbeck process for confinement.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_frames : int
        Number of frames.
    D : float
        Diffusion coefficient (µm²/s).
    confinement_radius : float
        Confinement radius in µm.
    dt : float
        Frame interval.
    dim : int
        Spatial dimensions.

    Returns
    -------
    ndarray
        Positions (n_particles, n_frames, dim).
    """
    sigma = np.sqrt(2 * D * dt)
    # Spring constant for OU process
    k = 2 * D / confinement_radius**2

    positions = np.zeros((n_particles, n_frames, dim))
    for t in range(1, n_frames):
        noise = np.random.randn(n_particles, dim) * sigma
        # OU drift toward origin
        drift = -k * positions[:, t - 1, :] * dt
        positions[:, t] = positions[:, t - 1] + drift + noise
    return positions


def anomalous_diffusion(n_particles, n_frames, D, alpha, dt=0.03,
                        dim=3):
    """Simulate anomalous diffusion via fractional Brownian motion.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_frames : int
        Number of frames.
    D : float
        Generalized diffusion coefficient.
    alpha : float
        Anomalous exponent (< 1 subdiffusive, > 1 superdiffusive).
    dt : float
        Frame interval.
    dim : int
        Spatial dimensions.

    Returns
    -------
    ndarray
        Positions (n_particles, n_frames, dim).
    """
    # Generate fractional Gaussian noise using Hosking method
    # For efficiency, use spectral method (approximate)
    H = alpha / 2.0  # Hurst exponent

    positions = np.zeros((n_particles, n_frames, dim))
    sigma = np.sqrt(2 * D * dt**alpha)

    for d_idx in range(dim):
        for p in range(n_particles):
            # Spectral synthesis of fBm
            fgn = _fractional_gaussian_noise(n_frames, H)
            positions[p, :, d_idx] = np.cumsum(fgn) * sigma

    return positions


def _fractional_gaussian_noise(n, H):
    """Generate fractional Gaussian noise via spectral method."""
    # Autocovariance of fGn
    k = np.arange(n)
    gamma_k = 0.5 * (np.abs(k - 1)**(2 * H) - 2 * np.abs(k)**(2 * H) +
                      np.abs(k + 1)**(2 * H))
    # Spectral density via FFT
    f_gamma = np.real(np.fft.fft(np.concatenate([gamma_k,
                                                  gamma_k[-2:0:-1]])))
    f_gamma = np.clip(f_gamma, 0, None)
    # Generate noise in frequency domain
    n_ext = len(f_gamma)
    z = np.random.randn(n_ext) + 1j * np.random.randn(n_ext)
    w = np.fft.ifft(np.sqrt(f_gamma) * z)
    return np.real(w[:n])


def switching_diffusion(n_particles, n_frames, D_states,
                        transition_matrix, dt=0.03, dim=3):
    """Simulate diffusion with HMM state switching.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    n_frames : int
        Number of frames.
    D_states : array-like
        Diffusion coefficients for each state.
    transition_matrix : ndarray
        (n_states, n_states) transition probability matrix per frame.
    dt : float
        Frame interval.
    dim : int
        Spatial dimensions.

    Returns
    -------
    tuple of (ndarray, ndarray)
        positions (n_particles, n_frames, dim),
        states (n_particles, n_frames) integer state labels.
    """
    D_states = np.asarray(D_states)
    transition_matrix = np.asarray(transition_matrix)
    n_states = len(D_states)

    positions = np.zeros((n_particles, n_frames, dim))
    states = np.zeros((n_particles, n_frames), dtype=int)

    # Initialize states uniformly
    states[:, 0] = np.random.randint(0, n_states, n_particles)

    for t in range(1, n_frames):
        # Transition
        for p in range(n_particles):
            s = states[p, t - 1]
            states[p, t] = np.random.choice(n_states,
                                             p=transition_matrix[s])
        # Move according to current state
        for s in range(n_states):
            mask = states[:, t] == s
            n_in_state = mask.sum()
            if n_in_state > 0:
                sigma = np.sqrt(2 * D_states[s] * dt)
                step = np.random.randn(n_in_state, dim) * sigma
                positions[mask, t] = positions[mask, t - 1] + step

    return positions, states


# ---------------------------------------------------------------------------
# Calcium signaling
# ---------------------------------------------------------------------------

def calcium_transient(n_frames, amplitude=5.0, tau_rise=0.05,
                      tau_decay=0.5, dt=0.03, baseline=1.0,
                      onset_frame=None):
    """Generate a single calcium transient (dF/F trace).

    Parameters
    ----------
    n_frames : int
        Number of frames.
    amplitude : float
        Peak dF/F amplitude.
    tau_rise : float
        Rise time constant in seconds.
    tau_decay : float
        Decay time constant in seconds.
    dt : float
        Frame interval in seconds.
    baseline : float
        Baseline fluorescence.
    onset_frame : int or None
        Frame of transient onset; None = n_frames // 4.

    Returns
    -------
    ndarray
        Fluorescence trace (n_frames,).
    """
    if onset_frame is None:
        onset_frame = n_frames // 4
    t = np.arange(n_frames) * dt
    t0 = onset_frame * dt
    trace = np.full(n_frames, baseline)
    mask = t >= t0
    t_rel = t[mask] - t0
    transient = amplitude * (1 - np.exp(-t_rel / tau_rise)) * \
        np.exp(-t_rel / tau_decay)
    trace[mask] += transient
    return trace


def calcium_spike_train(n_frames, rate=0.5, amplitude_range=(2, 10),
                        tau_decay=0.5, dt=0.03, baseline=1.0):
    """Generate calcium spike train with Poisson timing.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    rate : float
        Mean spike rate in Hz.
    amplitude_range : tuple
        (min, max) dF/F amplitude.
    tau_decay : float
        Decay time constant in seconds.
    dt : float
        Frame interval.
    baseline : float
        Baseline fluorescence.

    Returns
    -------
    ndarray
        Fluorescence trace (n_frames,).
    """
    trace = np.full(n_frames, baseline)
    tau_rise = 0.02  # fast rise

    # Generate spike times
    p_spike = rate * dt
    spikes = np.random.random(n_frames) < p_spike
    spike_frames = np.where(spikes)[0]

    for sf in spike_frames:
        amp = np.random.uniform(*amplitude_range)
        t_rel = (np.arange(n_frames) - sf) * dt
        mask = t_rel >= 0
        transient = amp * (1 - np.exp(-t_rel[mask] / tau_rise)) * \
            np.exp(-t_rel[mask] / tau_decay)
        trace[mask] += transient

    return trace


def calcium_wave(shape_3d, n_frames, origin, speed=10.0,
                 decay_length=50.0, dt=0.03, amplitude=5.0):
    """Generate a propagating calcium wave.

    Parameters
    ----------
    shape_3d : tuple
        (D, H, W) or (H, W) spatial shape.
    n_frames : int
        Number of frames.
    origin : tuple
        Wave origin coordinates.
    speed : float
        Wave speed in pixels/second.
    decay_length : float
        Spatial decay length in pixels.
    dt : float
        Frame interval.
    amplitude : float
        Peak amplitude.

    Returns
    -------
    ndarray
        (n_frames, *shape_3d) spatiotemporal calcium wave.
    """
    ndim = len(shape_3d)
    coords = np.ogrid[tuple(slice(0, s) for s in shape_3d)]
    dist = np.sqrt(sum((c - o)**2 for c, o in zip(coords, origin)))

    result = np.zeros((n_frames,) + shape_3d)
    for t in range(n_frames):
        wavefront = speed * t * dt
        # Wavefront profile: sharp rise, exponential decay behind
        behind = wavefront - dist
        profile = np.where(behind >= 0,
                           amplitude * np.exp(-behind / decay_length),
                           0.0)
        result[t] = profile
    return result


# ---------------------------------------------------------------------------
# DNA-PAINT
# ---------------------------------------------------------------------------

def dnapaint_binding_events(n_sites, n_frames, k_on=0.1, k_off=5.0,
                            imager_conc=1.0, dt=0.03):
    """Simulate DNA-PAINT transient binding events.

    Parameters
    ----------
    n_sites : int
        Number of binding sites.
    n_frames : int
        Number of frames.
    k_on : float
        Binding rate (s^-1 nM^-1).
    k_off : float
        Unbinding rate (s^-1).
    imager_conc : float
        Imager strand concentration in nM.
    dt : float
        Frame interval.

    Returns
    -------
    ndarray
        (n_frames, n_sites) binary binding state.
    """
    effective_on = k_on * imager_conc
    p_bind = 1 - np.exp(-effective_on * dt)
    p_unbind = 1 - np.exp(-k_off * dt)

    states = np.zeros((n_frames, n_sites), dtype=bool)
    bound = np.zeros(n_sites, dtype=bool)

    for t in range(n_frames):
        # Unbind
        unbind = np.random.random(n_sites) < p_unbind
        bound[bound & unbind] = False
        # Bind
        bind = np.random.random(n_sites) < p_bind
        bound[~bound & bind] = True
        states[t] = bound

    return states


# ---------------------------------------------------------------------------
# FLIM
# ---------------------------------------------------------------------------

def flim_decay(t, amplitude, lifetime, irf_sigma=0.05):
    """Generate FLIM fluorescence decay curve.

    Parameters
    ----------
    t : ndarray
        Time points in ns.
    amplitude : float
        Decay amplitude.
    lifetime : float
        Fluorescence lifetime in ns.
    irf_sigma : float
        Instrument response function width (ns).

    Returns
    -------
    ndarray
        Decay curve.
    """
    from scipy.special import erfc
    # Exponential decay convolved with Gaussian IRF
    # Analytical: A * exp(-t/tau) convolved with N(0, sigma)
    decay = amplitude * 0.5 * np.exp(
        irf_sigma**2 / (2 * lifetime**2) - t / lifetime
    ) * erfc(
        (irf_sigma**2 - t * lifetime) / (np.sqrt(2) * irf_sigma * lifetime)
    )
    return np.clip(decay, 0, None)


def flim_image(structure, lifetimes_map, n_photons=1000,
               time_bins=256, time_range=25.0):
    """Generate FLIM TCSPC histogram image.

    Parameters
    ----------
    structure : ndarray
        2D intensity structure (H, W).
    lifetimes_map : ndarray
        2D lifetime map in ns (H, W), or scalar.
    n_photons : int
        Mean photons per bright pixel.
    time_bins : int
        Number of time bins.
    time_range : float
        Time range in ns.

    Returns
    -------
    ndarray
        (H, W, time_bins) TCSPC histogram image.
    """
    h, w = structure.shape[:2]
    t = np.linspace(0, time_range, time_bins)

    if np.isscalar(lifetimes_map):
        lifetimes_map = np.full((h, w), lifetimes_map)

    result = np.zeros((h, w, time_bins))
    for yi in range(h):
        for xi in range(w):
            intensity = structure[yi, xi]
            if intensity <= 0:
                continue
            tau = lifetimes_map[yi, xi]
            if tau <= 0:
                continue
            # Generate decay histogram
            decay = flim_decay(t, intensity, tau)
            total_photons = int(n_photons * intensity)
            if total_photons > 0:
                # Sample from decay distribution
                prob = decay / (decay.sum() + 1e-10)
                counts = np.random.multinomial(total_photons, prob)
                result[yi, xi] = counts

    return result
