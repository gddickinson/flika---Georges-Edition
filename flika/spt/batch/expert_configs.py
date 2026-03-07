"""Pre-tuned SPT parameter profiles for common experiments.

Each profile provides an :class:`~flika.spt.batch.batch_pipeline.SPTParams`
instance optimised for a specific class of single-particle tracking
experiment.  Users can start from an expert configuration and tweak
individual parameters as needed.

Available profiles:

- **fast_membrane_proteins** -- fast-diffusing membrane receptors
  (TIRF, 20 Hz, 108 nm/px).
- **slow_confined_proteins** -- slowly diffusing or confined membrane
  proteins (TIRF, 20 Hz).
- **vesicle_trafficking** -- intracellular vesicle transport (widefield
  or confocal, longer gaps expected).
- **single_molecule** -- low-density single-molecule imaging (PALM/STORM
  style, high precision required).
- **viral_particles** -- virus particle tracking on cell membranes
  (intermediate speed, moderate density).
- **motor_proteins** -- directed transport by molecular motors (linear
  motion model preferred).

Usage::

    from flika.spt.batch.expert_configs import get_config, list_configs

    params = get_config('fast_membrane_proteins')
    params.pixel_size = 130.0  # override for your microscope
"""
from .batch_pipeline import SPTParams
from ...logger import logger


# ---------------------------------------------------------------------------
# Expert configuration profiles
# ---------------------------------------------------------------------------

EXPERT_CONFIGS = {
    'fast_membrane_proteins': SPTParams(
        detector='utrack',
        linker='greedy',
        max_distance=3.0,
        max_gap=36,
        min_track_length=36,
        psf_sigma=1.6,
        alpha=0.01,
        pixel_size=108.0,
        frame_interval=0.05,
        enable_features=True,
        enable_classification=False,
        enable_autocorrelation=False,
    ),

    'slow_confined_proteins': SPTParams(
        detector='utrack',
        linker='greedy',
        max_distance=2.0,
        max_gap=5,
        min_track_length=36,
        psf_sigma=1.6,
        alpha=0.01,
        pixel_size=108.0,
        frame_interval=0.05,
        enable_features=True,
        enable_classification=True,
        enable_autocorrelation=True,
        autocorrelation_intervals=15,
        autocorrelation_min_length=20,
    ),

    'vesicle_trafficking': SPTParams(
        detector='utrack',
        linker='utrack_lap',
        max_distance=10.0,
        max_gap=10,
        min_track_length=10,
        psf_sigma=2.0,
        alpha=0.05,
        pixel_size=108.0,
        frame_interval=0.1,
        enable_features=True,
        enable_classification=True,
        enable_autocorrelation=True,
        autocorrelation_intervals=20,
        autocorrelation_min_length=10,
    ),

    'single_molecule': SPTParams(
        detector='thunderstorm',
        linker='greedy',
        max_distance=5.0,
        max_gap=3,
        min_track_length=5,
        psf_sigma=1.3,
        alpha=0.01,
        pixel_size=108.0,
        frame_interval=0.05,
        enable_features=True,
        enable_classification=False,
        enable_autocorrelation=False,
        thunderstorm_filter='wavelet',
        thunderstorm_fitter='gaussian_mle',
        threshold=2.0,
    ),

    'viral_particles': SPTParams(
        detector='utrack',
        linker='utrack_lap',
        max_distance=8.0,
        max_gap=5,
        min_track_length=15,
        psf_sigma=1.8,
        alpha=0.05,
        pixel_size=108.0,
        frame_interval=0.05,
        enable_features=True,
        enable_classification=True,
        enable_autocorrelation=True,
        autocorrelation_intervals=20,
        autocorrelation_min_length=10,
    ),

    'motor_proteins': SPTParams(
        detector='utrack',
        linker='utrack_lap',
        max_distance=12.0,
        max_gap=3,
        min_track_length=20,
        psf_sigma=1.5,
        alpha=0.05,
        pixel_size=108.0,
        frame_interval=0.1,
        enable_features=True,
        enable_classification=False,
        enable_autocorrelation=True,
        autocorrelation_intervals=30,
        autocorrelation_min_length=15,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config(name):
    """Get an expert configuration by name.

    Returns a *copy* of the stored :class:`SPTParams` so that
    modifications do not affect the template.

    Args:
        name: Configuration profile name (see :data:`EXPERT_CONFIGS`).

    Returns:
        :class:`SPTParams` instance.

    Raises:
        KeyError: If *name* is not a recognised profile.
    """
    if name not in EXPERT_CONFIGS:
        available = ', '.join(sorted(EXPERT_CONFIGS.keys()))
        raise KeyError(
            f"Unknown expert config {name!r}. "
            f"Available configs: {available}")

    # Return a copy via dict round-trip
    template = EXPERT_CONFIGS[name]
    return SPTParams.from_dict(template.to_dict())


def list_configs():
    """List all available expert configuration names.

    Returns:
        list of str, sorted alphabetically.
    """
    return sorted(EXPERT_CONFIGS.keys())


def describe_config(name):
    """Return a human-readable summary of a configuration.

    Args:
        name: Configuration profile name.

    Returns:
        str with a multi-line description of the parameters.

    Raises:
        KeyError: If *name* is not a recognised profile.
    """
    config = get_config(name)
    params = config.to_dict()

    lines = [f"Expert config: {name}", "=" * (len(name) + 16)]
    for key, value in sorted(params.items()):
        lines.append(f"  {key}: {value!r}")
    return '\n'.join(lines)


def get_config_for_experiment(particle_type=None, speed=None,
                              density=None, microscope=None):
    """Suggest an expert config based on experiment characteristics.

    This is a simple heuristic lookup.  For more precise tuning, start
    from the suggested config and adjust parameters.

    Args:
        particle_type: ``'membrane_protein'``, ``'vesicle'``,
            ``'single_molecule'``, ``'virus'``, ``'motor'``, or ``None``.
        speed: ``'fast'``, ``'slow'``, ``'medium'``, or ``None``.
        density: ``'low'``, ``'medium'``, ``'high'``, or ``None``.
        microscope: ``'tirf'``, ``'widefield'``, ``'confocal'``,
            ``'palm_storm'``, or ``None``.

    Returns:
        (config_name, SPTParams) tuple.
    """
    # Direct particle type mapping
    if particle_type == 'vesicle':
        return 'vesicle_trafficking', get_config('vesicle_trafficking')
    if particle_type == 'single_molecule':
        return 'single_molecule', get_config('single_molecule')
    if particle_type == 'virus':
        return 'viral_particles', get_config('viral_particles')
    if particle_type == 'motor':
        return 'motor_proteins', get_config('motor_proteins')

    # Microscope-based suggestion
    if microscope == 'palm_storm':
        return 'single_molecule', get_config('single_molecule')

    # Speed-based suggestion for membrane proteins
    if speed == 'fast' or (particle_type == 'membrane_protein' and speed != 'slow'):
        return 'fast_membrane_proteins', get_config('fast_membrane_proteins')
    if speed == 'slow':
        return 'slow_confined_proteins', get_config('slow_confined_proteins')

    # Default
    name = 'fast_membrane_proteins'
    logger.info("No specific match; suggesting default config: %s", name)
    return name, get_config(name)
