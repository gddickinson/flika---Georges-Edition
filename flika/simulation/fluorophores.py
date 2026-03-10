# -*- coding: utf-8 -*-
"""Fluorophore and labeling models for microscopy simulation."""
import numpy as np
from dataclasses import dataclass


@dataclass
class FluorophoreConfig:
    """Fluorophore photophysical properties.

    Parameters
    ----------
    name : str
        Fluorophore name.
    wavelength_ex : float
        Excitation wavelength in nm.
    wavelength_em : float
        Emission wavelength in nm.
    extinction_coeff : float
        Molar extinction coefficient (M^-1 cm^-1).
    quantum_yield : float
        Fluorescence quantum yield (0-1).
    photons_per_frame : float
        Mean detected photons per molecule per frame.
    on_rate : float
        Switching on rate (s^-1) for blinking; 0 = always on.
    off_rate : float
        Switching off rate (s^-1) for blinking; 0 = always on.
    bleach_rate : float
        Photobleaching rate (s^-1); 0 = no bleaching.
    lifetime : float
        Fluorescence lifetime in ns (for FLIM).
    """
    name: str = 'Generic'
    wavelength_ex: float = 488.0
    wavelength_em: float = 520.0
    extinction_coeff: float = 73000
    quantum_yield: float = 0.92
    photons_per_frame: float = 2000
    on_rate: float = 0.0
    off_rate: float = 0.0
    bleach_rate: float = 0.0
    lifetime: float = 3.5


FLUOROPHORE_PRESETS = {
    'EGFP': FluorophoreConfig(
        'EGFP', 488, 507, 56000, 0.60, 1500,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.001, lifetime=2.6),
    'mCherry': FluorophoreConfig(
        'mCherry', 587, 610, 72000, 0.22, 800,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.002, lifetime=1.4),
    'Alexa488': FluorophoreConfig(
        'Alexa488', 495, 519, 73000, 0.92, 3000,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.0005, lifetime=4.1),
    'Alexa647': FluorophoreConfig(
        'Alexa647', 650, 668, 270000, 0.33, 5000,
        on_rate=0.1, off_rate=50.0, bleach_rate=0.0001, lifetime=1.0),
    'Atto655': FluorophoreConfig(
        'Atto655', 663, 684, 125000, 0.30, 4000,
        on_rate=0.05, off_rate=20.0, bleach_rate=0.0002, lifetime=1.8),
    'GCaMP6f': FluorophoreConfig(
        'GCaMP6f', 488, 510, 56000, 0.59, 1200,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.0005, lifetime=2.7),
    'GCaMP6s': FluorophoreConfig(
        'GCaMP6s', 488, 510, 56000, 0.59, 1400,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.0005, lifetime=2.7),
    'DAPI': FluorophoreConfig(
        'DAPI', 360, 460, 27000, 0.58, 1000,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.0001, lifetime=2.8),
    'Hoechst': FluorophoreConfig(
        'Hoechst', 350, 461, 40000, 0.60, 1100,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.0001, lifetime=2.5),
    'MitoTracker': FluorophoreConfig(
        'MitoTracker', 579, 599, 112000, 0.30, 2000,
        on_rate=0.0, off_rate=0.0, bleach_rate=0.001, lifetime=3.2),
}


def label_structure(structure, density, fluorophore=None):
    """Place fluorophore labels on a binary structure.

    Parameters
    ----------
    structure : ndarray
        Binary structure mask.
    density : float
        Labels per pixel on the structure (probability for each
        structure voxel).
    fluorophore : FluorophoreConfig or None
        Fluorophore to assign. Defaults to generic.

    Returns
    -------
    list of dict
        Each dict has 'position' (tuple) and 'fluorophore' (config).
    """
    if fluorophore is None:
        fluorophore = FluorophoreConfig()

    # Get all structure voxel positions
    positions = np.argwhere(structure > 0)
    if len(positions) == 0:
        return []

    # Randomly select based on density
    density = np.clip(density, 0, 1)
    mask = np.random.random(len(positions)) < density
    selected = positions[mask]

    labels = []
    for pos in selected:
        # Add sub-pixel offset for realism
        offset = np.random.uniform(-0.5, 0.5, size=len(pos))
        labels.append({
            'position': tuple(pos.astype(float) + offset),
            'fluorophore': fluorophore,
        })
    return labels


def simulate_blinking(n_frames, on_rate, off_rate, bleach_rate=0.0,
                      dt=0.03, initial_state=True):
    """Simulate fluorophore blinking via Markov chain.

    Parameters
    ----------
    n_frames : int
        Number of time frames.
    on_rate : float
        Off→On transition rate (s^-1).
    off_rate : float
        On→Off transition rate (s^-1).
    bleach_rate : float
        On→Bleached rate (s^-1).
    dt : float
        Frame interval in seconds.
    initial_state : bool
        Starting state (True = on).

    Returns
    -------
    ndarray
        Boolean array of shape (n_frames,), True = emitting.
    """
    states = np.zeros(n_frames, dtype=bool)
    state = initial_state
    bleached = False

    # Transition probabilities per frame
    p_on_to_off = 1 - np.exp(-off_rate * dt) if off_rate > 0 else 0
    p_off_to_on = 1 - np.exp(-on_rate * dt) if on_rate > 0 else 0
    p_bleach = 1 - np.exp(-bleach_rate * dt) if bleach_rate > 0 else 0

    for t in range(n_frames):
        if bleached:
            states[t] = False
            continue
        states[t] = state
        if state:
            # On → check bleach first
            if np.random.random() < p_bleach:
                bleached = True
                continue
            if np.random.random() < p_on_to_off:
                state = False
        else:
            if np.random.random() < p_off_to_on:
                state = True

    return states


def simulate_bleaching(n_frames, bleach_rate, initial_intensity=1.0,
                       dt=0.03):
    """Simulate photobleaching as exponential decay with noise.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    bleach_rate : float
        Bleaching rate constant (s^-1).
    initial_intensity : float
        Starting intensity.
    dt : float
        Frame interval in seconds.

    Returns
    -------
    ndarray
        Intensity values of shape (n_frames,).
    """
    t = np.arange(n_frames) * dt
    intensity = initial_intensity * np.exp(-bleach_rate * t)
    return intensity


def simulate_blinking_batch(n_emitters, n_frames, on_rate, off_rate,
                            bleach_rate=0.0, dt=0.03):
    """Simulate blinking for multiple emitters efficiently.

    Parameters
    ----------
    n_emitters : int
        Number of emitters.
    n_frames : int
        Number of frames.
    on_rate, off_rate, bleach_rate : float
        Rate constants.
    dt : float
        Frame interval.

    Returns
    -------
    ndarray
        Boolean array (n_emitters, n_frames).
    """
    states = np.zeros((n_emitters, n_frames), dtype=bool)

    p_on_to_off = 1 - np.exp(-off_rate * dt) if off_rate > 0 else 0
    p_off_to_on = 1 - np.exp(-on_rate * dt) if on_rate > 0 else 0
    p_bleach = 1 - np.exp(-bleach_rate * dt) if bleach_rate > 0 else 0

    # If no blinking, all on with bleaching only
    if on_rate == 0 and off_rate == 0:
        if bleach_rate > 0:
            for i in range(n_emitters):
                bleach_frame = np.random.geometric(p_bleach) if p_bleach > 0 else n_frames
                states[i, :min(bleach_frame, n_frames)] = True
        else:
            states[:] = True
        return states

    # Full Markov simulation
    current = np.ones(n_emitters, dtype=bool)
    bleached = np.zeros(n_emitters, dtype=bool)
    rng = np.random.random

    for t in range(n_frames):
        active = ~bleached
        states[active, t] = current[active]

        # Bleach check for on-state emitters
        if p_bleach > 0:
            on_mask = current & active
            bleach_mask = rng(n_emitters) < p_bleach
            bleached[on_mask & bleach_mask] = True

        # On → Off
        if p_on_to_off > 0:
            on_mask = current & ~bleached
            switch_off = rng(n_emitters) < p_on_to_off
            current[on_mask & switch_off] = False

        # Off → On
        if p_off_to_on > 0:
            off_mask = ~current & ~bleached
            switch_on = rng(n_emitters) < p_off_to_on
            current[off_mask & switch_on] = True

    return states
