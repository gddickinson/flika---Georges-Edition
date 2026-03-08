# -*- coding: utf-8 -*-
"""Named simulation presets for quick access."""
from .engine import SimulationConfig
from .noise import CameraConfig


PRESETS = {
    'Beads - PSF Calibration': SimulationConfig(
        nx=256, ny=256, nz=1, nt=1,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 100, 'bead_radius': 0.1},
        fluorophore='Alexa488',
        camera=CameraConfig(type='sCMOS'),
    ),

    'TIRF - Single Molecules': SimulationConfig(
        nx=512, ny=512, nz=10, nt=100,
        modality='tirf', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 200, 'bead_radius': 0.1},
        fluorophore='Alexa647',
        camera=CameraConfig(type='EMCCD', em_gain=300,
                             read_noise=0.5, quantum_efficiency=0.95),
        modality_params={'penetration_depth': 100},
    ),

    'Confocal - Cell with Organelles': SimulationConfig(
        nx=512, ny=512, nz=30, nt=1,
        pixel_size=0.08, z_step=0.3,
        modality='confocal', psf_model='gaussian',
        structure_type='cells',
        structure_params={'cell_type': 'round', 'organelles': True},
        fluorophore='EGFP',
        camera=CameraConfig(type='sCMOS'),
        modality_params={'pinhole_au': 1.0},
    ),

    'PALM/STORM - Dense Filaments': SimulationConfig(
        nx=256, ny=256, nz=1, nt=5000,
        pixel_size=0.1, dt=0.03,
        modality='smlm', psf_model='gaussian',
        structure_type='filaments',
        structure_params={'n_filaments': 20, 'persistence_length': 30},
        fluorophore='Alexa647',
        camera=CameraConfig(type='EMCCD', em_gain=300),
        modality_params={'density_per_frame': 2.0},
    ),

    'DNA-PAINT - Grid Pattern': SimulationConfig(
        nx=256, ny=256, nz=1, nt=2000,
        pixel_size=0.1, dt=0.1,
        modality='dnapaint', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 100, 'bead_radius': 0.1},
        fluorophore='Atto655',
        camera=CameraConfig(type='EMCCD', em_gain=300),
        modality_params={'k_on': 0.1, 'k_off': 5.0, 'imager_conc': 1.0},
    ),

    'Light-Sheet - Cell Spheroid': SimulationConfig(
        nx=256, ny=256, nz=50, nt=1,
        pixel_size=0.3, z_step=1.0,
        modality='lightsheet', psf_model='gaussian',
        structure_type='cell_field',
        structure_params={'n_cells': 10, 'cell_type': 'round'},
        fluorophore='EGFP',
        camera=CameraConfig(type='sCMOS'),
        modality_params={'sheet_thickness': 2.0},
    ),

    'FLIM - Two Lifetimes': SimulationConfig(
        nx=128, ny=128, nz=1, nt=1,
        modality='flim', psf_model='gaussian',
        structure_type='cells',
        structure_params={'cell_type': 'round'},
        fluorophore='EGFP',
        camera=CameraConfig(type='sCMOS'),
        modality_params={'time_bins': 256, 'time_range': 25.0,
                         'n_photons': 1000},
    ),

    'Calcium - Neuron Spikes': SimulationConfig(
        nx=128, ny=128, nz=1, nt=1000,
        pixel_size=0.5, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='cells',
        structure_params={'cell_type': 'neuron'},
        fluorophore='GCaMP6f',
        modality_params={'calcium_rate': 0.5, 'calcium_tau': 0.5},
        camera=CameraConfig(type='sCMOS'),
    ),

    'SIM - Resolution Target': SimulationConfig(
        nx=256, ny=256, nz=1, nt=1,
        pixel_size=0.05,
        modality='sim', psf_model='gaussian',
        structure_type='filaments',
        structure_params={'n_filaments': 15, 'persistence_length': 50,
                         'thickness': 0.5},
        fluorophore='Alexa488',
        camera=CameraConfig(type='sCMOS'),
    ),

    'Tracking - Brownian Particles': SimulationConfig(
        nx=256, ny=256, nz=1, nt=500,
        pixel_size=0.1, dt=0.03,
        modality='widefield', psf_model='gaussian',
        structure_type='beads',
        structure_params={'n_beads': 50, 'bead_radius': 0.1},
        fluorophore='EGFP',
        motion_type='brownian',
        motion_params={'D': 0.1},
        camera=CameraConfig(type='sCMOS'),
    ),
}
