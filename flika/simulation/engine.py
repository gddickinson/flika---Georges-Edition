# -*- coding: utf-8 -*-
"""Simulation pipeline coordinator."""
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import fftconvolve

from .noise import CameraConfig


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    Parameters
    ----------
    nx, ny : int
        Image width and height in pixels.
    nz : int
        Number of z-slices (1 for 2D).
    nt : int
        Number of time frames (1 for static).
    pixel_size : float
        Pixel size in µm.
    z_step : float
        Z step in µm.
    dt : float
        Frame interval in seconds.
    modality : str
        Microscope modality.
    wavelength : float
        Emission wavelength in nm.
    NA : float
        Numerical aperture.
    n_immersion : float
        Immersion medium refractive index.
    psf_model : str
        PSF model name.
    structure_type : str
        Sample structure type.
    structure_params : dict
        Structure-specific parameters.
    fluorophore : str
        Fluorophore preset name or 'custom'.
    labeling_density : float
        Fluorophore labeling density (0-1).
    camera : CameraConfig
        Camera configuration.
    motion_type : str
        Particle dynamics type.
    motion_params : dict
        Motion-specific parameters.
    modality_params : dict
        Modality-specific parameters.
    """
    nx: int = 256
    ny: int = 256
    nz: int = 1
    nt: int = 1
    pixel_size: float = 0.1
    z_step: float = 0.3
    dt: float = 0.03

    modality: str = 'widefield'
    wavelength: float = 520.0
    NA: float = 1.4
    n_immersion: float = 1.515
    psf_model: str = 'gaussian'

    structure_type: str = 'beads'
    structure_params: dict = field(default_factory=dict)

    fluorophore: str = 'EGFP'
    labeling_density: float = 1.0

    camera: CameraConfig = field(default_factory=CameraConfig)

    motion_type: str = 'static'
    motion_params: dict = field(default_factory=dict)

    modality_params: dict = field(default_factory=dict)


class SimulationEngine:
    """Composable microscopy simulation pipeline.

    Pipeline: Structure -> Fluorophore labeling -> PSF -> Dynamics ->
              Optics -> Camera/Noise -> Output
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : SimulationConfig
            Complete simulation configuration.
        """
        self.config = config

    def run(self, progress_callback=None):
        """Execute simulation pipeline.

        Parameters
        ----------
        progress_callback : callable or None
            Called with (percent, message) during execution.

        Returns
        -------
        tuple of (ndarray, dict)
            (image_stack, ground_truth_metadata)
        """
        cfg = self.config
        gt = {}  # ground truth
        self._extra_gt = {}  # populated by sub-methods

        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        _progress(5, "Generating structure...")
        structure, positions = self._generate_structure()
        gt['structure_mask'] = structure.copy()
        gt['config'] = cfg

        _progress(15, "Generating PSF...")
        psf = self._generate_psf()

        _progress(25, "Setting up fluorophores...")
        from .fluorophores import FLUOROPHORE_PRESETS, FluorophoreConfig
        if cfg.fluorophore in FLUOROPHORE_PRESETS:
            fluor = FLUOROPHORE_PRESETS[cfg.fluorophore]
        else:
            fluor = FluorophoreConfig()

        # Determine output shape
        is_3d = cfg.nz > 1
        spatial = (cfg.nz, cfg.ny, cfg.nx) if is_3d else (cfg.ny, cfg.nx)

        _progress(30, "Simulating dynamics...")
        trajectories = None
        if cfg.motion_type != 'static' and positions is not None and cfg.nt > 1:
            trajectories = self._simulate_dynamics(positions)
            gt['trajectories'] = trajectories

        _progress(40, "Rendering frames...")
        if positions is not None:
            gt['positions'] = positions

        # Enhanced GT: per-frame detection table and track dict
        if positions is not None:
            from .ground_truth import (positions_to_detection_table,
                                       trajectories_to_track_dict)
            gt['positions_per_frame'] = positions_to_detection_table(
                positions, trajectories, cfg)
            if trajectories is not None:
                gt['track_dict'] = trajectories_to_track_dict(
                    positions, trajectories, cfg)
                # Store true diffusion coefficient if available
                D = cfg.motion_params.get('D')
                if D is not None:
                    gt['diffusion_coefficients'] = np.full(
                        len(positions), D)

        # Enhanced GT: segmentation labels for cell fields
        if cfg.structure_type == 'cell_field':
            from . import structures as S2
            sp = cfg.structure_params
            label_shape = (cfg.nz, cfg.ny, cfg.nx) if is_3d \
                else (cfg.ny, cfg.nx)
            gt['segmentation_mask'] = S2.generate_cell_field(
                label_shape,
                n_cells=sp.get('n_cells', 20),
                cell_type=sp.get('cell_type', 'round'),
                labels=True)

        # Binary mask for any structure
        gt['binary_mask'] = (structure > 0).astype(np.uint8)

        # Generate time series
        stack = self._render_stack(structure, positions, trajectories,
                                   psf, fluor, _progress)

        _progress(85, "Applying camera model...")
        from .noise import apply_camera, apply_background
        bg_level = cfg.modality_params.get('background', 10)
        if bg_level > 0:
            stack = apply_background(stack, bg_level)

        gt['n_photons'] = stack.copy()
        gt.update(self._extra_gt)
        stack = apply_camera(stack.astype(float), cfg.camera)

        # Reshape to match flika's dimension conventions:
        #   2D -> (Y, X)
        #   3D time series -> (T, Y, X)
        #   3D z-stack (static) -> (1, Y, X, Z)  — Z slider via dim-3
        #   4D time+z -> (T, Y, X, Z)  — time slider + Z slider
        # Internally the engine uses z-first: (Z, Y, X) or (T, Z, Y, X).
        # Flika expects z-last for the extra-dimension slider system.
        if stack.ndim == 3 and cfg.nz > 1 and cfg.nt <= 1:
            # Static z-stack (Z, Y, X) -> (1, Y, X, Z)
            stack = np.transpose(stack, (1, 2, 0))[np.newaxis, ...]
        elif stack.ndim == 4 and cfg.nz > 1:
            # Time + z  (T, Z, Y, X) -> (T, Y, X, Z)
            stack = np.transpose(stack, (0, 2, 3, 1))

        _progress(100, "Done.")
        return stack, gt

    def _generate_structure(self):
        """Generate sample structure and emitter positions."""
        cfg = self.config
        is_3d = cfg.nz > 1
        shape = (cfg.nz, cfg.ny, cfg.nx) if is_3d else (cfg.ny, cfg.nx)
        params = cfg.structure_params

        from . import structures as S

        positions = None

        if cfg.structure_type == 'beads':
            n_beads = params.get('n_beads', 100)
            bead_radius = params.get('bead_radius', 0.1)
            ndim = 3 if is_3d else 2
            pos = np.column_stack(
                [np.random.uniform(5, s - 5, n_beads)
                 for s in shape])
            structure = S.generate_bead_field(shape, n_beads, bead_radius,
                                             pos)
            positions = pos

        elif cfg.structure_type == 'filaments':
            n_fil = params.get('n_filaments', 10)
            persistence = params.get('persistence_length', 20.0)
            thickness = params.get('thickness', 1.0)
            structure = S.generate_filaments(
                shape, n_fil, persistence, thickness)

        elif cfg.structure_type == 'cells':
            cell_type = params.get('cell_type', 'round')
            structure = S.generate_cell_mask(shape, cell_type)
            organelles = params.get('organelles', False)
            if organelles:
                nuc = S.generate_nucleus(structure, 0.3)
                mito = S.generate_mitochondria(structure, 20)
                # Combine: nucleus bright, mito medium, cytoplasm dim
                combined = structure * 0.2
                combined[nuc > 0] = 1.0
                combined[mito > 0] = 0.7
                structure = combined

        elif cfg.structure_type == 'cell_field':
            n_cells = params.get('n_cells', 20)
            cell_type = params.get('cell_type', 'round')
            structure = S.generate_cell_field(
                shape, n_cells, cell_type, labels=False)

        else:
            # Custom or unknown: empty structure with random points
            structure = np.zeros(shape)
            n_pts = params.get('n_points', 50)
            ndim = len(shape)
            for _ in range(n_pts):
                idx = tuple(np.random.randint(0, s) for s in shape)
                structure[idx] = 1.0

        return structure, positions

    def _generate_psf(self):
        """Generate PSF based on config."""
        cfg = self.config
        from . import psf as P

        # PSF size: ~4x Airy radius
        wl_um = cfg.wavelength / 1000.0
        airy_px = 0.61 * wl_um / cfg.NA / cfg.pixel_size
        psf_size = max(7, int(airy_px * 6) | 1)  # ensure odd

        if cfg.psf_model == 'gaussian':
            sigma = 0.21 * wl_um / cfg.NA / cfg.pixel_size
            if cfg.nz > 1:
                sigma_z = 0.66 * wl_um / (cfg.n_immersion -
                          np.sqrt(cfg.n_immersion**2 - cfg.NA**2))
                sigma_z_px = sigma_z / cfg.z_step
                psf_d = max(5, int(sigma_z_px * 6) | 1)
                return P.gaussian_psf_3d((psf_d, psf_size, psf_size),
                                         sigma, sigma_z_px)
            else:
                return P.gaussian_psf_2d((psf_size, psf_size), sigma)

        elif cfg.psf_model == 'airy':
            return P.airy_psf_2d((psf_size, psf_size), cfg.wavelength,
                                 cfg.NA, cfg.pixel_size)

        elif cfg.psf_model == 'born_wolf':
            psf_d = max(5, psf_size // 2) | 1
            return P.born_wolf_psf_3d(
                (psf_d, psf_size, psf_size),
                cfg.wavelength, cfg.NA, cfg.n_immersion,
                cfg.pixel_size, cfg.z_step)

        elif cfg.psf_model == 'vectorial':
            psf_d = max(5, psf_size // 2) | 1
            return P.vectorial_psf_3d(
                (psf_d, psf_size, psf_size),
                cfg.wavelength, cfg.NA, cfg.n_immersion,
                cfg.pixel_size, cfg.z_step)
        else:
            # Default to Gaussian
            sigma = 0.21 * wl_um / cfg.NA / cfg.pixel_size
            return P.gaussian_psf_2d((psf_size, psf_size), sigma)

    def _simulate_dynamics(self, positions):
        """Simulate particle dynamics."""
        cfg = self.config
        from . import dynamics as D

        n_particles = len(positions)
        dim = 3 if cfg.nz > 1 else 2
        mp = cfg.motion_params

        if cfg.motion_type == 'brownian':
            diff_coeff = mp.get('D', 0.1)
            traj = D.brownian_motion(n_particles, cfg.nt, diff_coeff,
                                     cfg.dt, dim)
        elif cfg.motion_type == 'directed':
            diff_coeff = mp.get('D', 0.05)
            vel = mp.get('velocity', [0.5, 0, 0])
            traj = D.directed_motion(n_particles, cfg.nt, diff_coeff,
                                     vel, cfg.dt, dim)
        elif cfg.motion_type == 'confined':
            diff_coeff = mp.get('D', 0.1)
            conf_r = mp.get('confinement_radius', 1.0)
            traj = D.confined_motion(n_particles, cfg.nt, diff_coeff,
                                     conf_r, cfg.dt, dim)
        elif cfg.motion_type == 'anomalous':
            diff_coeff = mp.get('D', 0.1)
            alpha = mp.get('alpha', 0.5)
            traj = D.anomalous_diffusion(n_particles, cfg.nt,
                                         diff_coeff, alpha, cfg.dt, dim)
        else:
            return None

        # Convert from µm to pixels
        traj[:, :, -1] /= cfg.pixel_size  # x
        traj[:, :, -2] /= cfg.pixel_size  # y
        if dim == 3:
            traj[:, :, 0] /= cfg.z_step  # z

        return traj

    def _render_stack(self, structure, positions, trajectories, psf,
                      fluor, _progress):
        """Render the full image stack."""
        cfg = self.config
        is_3d = cfg.nz > 1

        if cfg.modality == 'smlm':
            return self._render_smlm(structure, positions, psf, fluor,
                                     _progress)
        elif cfg.modality == 'dnapaint':
            return self._render_dnapaint(structure, positions, psf,
                                         fluor, _progress)
        elif cfg.modality == 'flim':
            return self._render_flim(structure, fluor, _progress)

        spatial = (cfg.nz, cfg.ny, cfg.nx) if is_3d else (cfg.ny, cfg.nx)

        if cfg.nt <= 1:
            # Static image
            photon_img = self._render_single_frame(
                structure, positions, psf, fluor)
            photon_img = self._apply_modality(photon_img)
            # Squeeze out z if single plane
            if is_3d and photon_img.ndim == 3:
                return photon_img
            return photon_img

        # Time series
        if is_3d:
            stack = np.zeros((cfg.nt, cfg.nz, cfg.ny, cfg.nx))
        else:
            stack = np.zeros((cfg.nt, cfg.ny, cfg.nx))

        # Bleaching curve
        from .fluorophores import simulate_bleaching
        bleach = simulate_bleaching(cfg.nt, fluor.bleach_rate,
                                    dt=cfg.dt)

        # Calcium mode
        calcium_trace = None
        if cfg.modality_params.get('calcium_rate', 0) > 0:
            from .dynamics import calcium_spike_train
            calcium_trace = calcium_spike_train(
                cfg.nt,
                rate=cfg.modality_params.get('calcium_rate', 0.5),
                tau_decay=cfg.modality_params.get('calcium_tau', 0.5),
                dt=cfg.dt)
            self._extra_gt['calcium_trace'] = calcium_trace.copy()
            # Find spike frames (local maxima above baseline)
            baseline = np.median(calcium_trace)
            threshold = baseline + 2 * np.std(calcium_trace[:max(10, cfg.nt // 10)])
            peaks = []
            for i in range(1, len(calcium_trace) - 1):
                if (calcium_trace[i] > threshold and
                        calcium_trace[i] > calcium_trace[i - 1] and
                        calcium_trace[i] >= calcium_trace[i + 1]):
                    peaks.append(i)
            self._extra_gt['calcium_spike_frames'] = np.array(peaks)

        for t in range(cfg.nt):
            pct = 40 + int(45 * t / max(cfg.nt, 1))
            if t % max(1, cfg.nt // 20) == 0:
                _progress(pct, f"Rendering frame {t+1}/{cfg.nt}...")

            # Update positions for dynamics
            if trajectories is not None and positions is not None:
                current_pos = positions + trajectories[:, t, :]
                frame_struct = self._positions_to_image(
                    current_pos, spatial)
            else:
                frame_struct = structure

            # Apply bleaching
            frame_struct = frame_struct * bleach[t]

            # Apply calcium modulation
            if calcium_trace is not None:
                frame_struct = frame_struct * calcium_trace[t]

            # Convolve with PSF
            frame = self._render_single_frame(
                frame_struct, None, psf, fluor, skip_label=True)
            frame = self._apply_modality(frame)

            if is_3d:
                if frame.ndim == 3:
                    stack[t] = frame[:cfg.nz, :cfg.ny, :cfg.nx]
                else:
                    stack[t, 0] = frame[:cfg.ny, :cfg.nx]
            else:
                if frame.ndim == 2:
                    stack[t] = frame[:cfg.ny, :cfg.nx]
                elif frame.ndim == 3:
                    # Sum z for widefield
                    stack[t] = frame.sum(axis=0)[:cfg.ny, :cfg.nx]

        return stack

    def _render_single_frame(self, structure, positions, psf, fluor,
                             skip_label=False):
        """Render a single frame by convolving structure with PSF."""
        cfg = self.config
        # Scale structure by photon count
        photon_image = structure.astype(float) * fluor.photons_per_frame

        # Convolve with PSF
        if psf.ndim == photon_image.ndim:
            result = fftconvolve(photon_image, psf, mode='same')
        elif psf.ndim == 2 and photon_image.ndim == 3:
            # Apply 2D PSF per z-plane
            result = np.zeros_like(photon_image)
            for zi in range(photon_image.shape[0]):
                result[zi] = fftconvolve(photon_image[zi], psf,
                                         mode='same')
        elif psf.ndim == 3 and photon_image.ndim == 2:
            # Use central z-plane of 3D PSF
            cz = psf.shape[0] // 2
            result = fftconvolve(photon_image, psf[cz], mode='same')
        else:
            result = photon_image

        return np.clip(result, 0, None)

    def _apply_modality(self, photon_image):
        """Apply modality-specific optical effects."""
        cfg = self.config

        if cfg.modality == 'tirf':
            from .optics import tirf_excitation_profile
            depth = cfg.modality_params.get('penetration_depth', 100)
            if photon_image.ndim == 3:
                nz = photon_image.shape[0]
                z_nm = np.arange(nz) * cfg.z_step * 1000  # µm to nm
                profile = tirf_excitation_profile(z_nm, depth)
                for zi in range(nz):
                    photon_image[zi] *= profile[zi]
                # TIRF: return sum or bottom slice
                return photon_image[0]
            return photon_image

        elif cfg.modality == 'confocal':
            from .optics import confocal_scan
            if photon_image.ndim == 3:
                # Already convolved; apply pinhole sectioning
                pinhole = cfg.modality_params.get('pinhole_au', 1.0)
                from scipy.ndimage import gaussian_filter
                # Approximate confocal sectioning by reducing
                # out-of-focus contribution
                result = np.zeros_like(photon_image)
                for zi in range(photon_image.shape[0]):
                    result[zi] = photon_image[zi] / max(1, pinhole)
                return result
            return photon_image

        elif cfg.modality == 'lightsheet':
            if photon_image.ndim == 3:
                thickness = cfg.modality_params.get('sheet_thickness', 2.0)
                nz = photon_image.shape[0]
                result = np.zeros_like(photon_image)
                for zi in range(nz):
                    # Sheet profile centered at each z
                    z_profile = np.exp(-0.5 * ((np.arange(nz) - zi) /
                                                thickness)**2)
                    sectioned = (photon_image * z_profile[:, None, None]
                                 ).sum(axis=0)
                    result[zi] = sectioned
                return result
            return photon_image

        elif cfg.modality == 'sim':
            # Apply structured illumination patterns
            from .optics import sim_patterns, sim_reconstruct
            h, w = photon_image.shape[-2:]
            patterns = sim_patterns((h, w))
            # Generate raw SIM images
            if photon_image.ndim == 2:
                raw = photon_image[None, :, :] * patterns
                return sim_reconstruct(raw, patterns)
            return photon_image

        elif cfg.modality == 'sted':
            # STED effect already in PSF; just return
            return photon_image

        # widefield or default: no modification
        return photon_image

    def _render_smlm(self, structure, positions, psf, fluor, _progress):
        """Render SMLM (PALM/STORM) data."""
        cfg = self.config
        from .fluorophores import simulate_blinking_batch

        # Get emitter positions from structure
        if positions is None:
            positions = np.argwhere(structure > 0).astype(float)
            # Sub-pixel randomization
            positions += np.random.uniform(-0.5, 0.5, positions.shape)

        n_emitters = len(positions)
        if n_emitters == 0:
            return np.zeros((cfg.nt, cfg.ny, cfg.nx))

        # Simulate blinking
        states = simulate_blinking_batch(
            n_emitters, cfg.nt,
            fluor.on_rate, fluor.off_rate, fluor.bleach_rate, cfg.dt)

        # Density control
        density = cfg.modality_params.get('density_per_frame', 2.0)
        # Adjust on_rate to get desired density
        target_on_fraction = density / max(n_emitters, 1)
        if target_on_fraction < 1:
            # Further sparsify
            sparse_mask = np.random.random(states.shape) < target_on_fraction
            states = states & sparse_mask

        self._extra_gt['emitter_states'] = states

        stack = np.zeros((cfg.nt, cfg.ny, cfg.nx))
        psf_2d = psf if psf.ndim == 2 else psf[psf.shape[0] // 2]

        for t in range(cfg.nt):
            if t % max(1, cfg.nt // 20) == 0:
                pct = 40 + int(45 * t / max(cfg.nt, 1))
                _progress(pct, f"SMLM frame {t+1}/{cfg.nt}...")

            on_mask = states[:, t]
            on_pos = positions[on_mask]
            if len(on_pos) == 0:
                continue

            frame = np.zeros((cfg.ny, cfg.nx))
            for pos in on_pos:
                # Place PSF at each emitter position
                if len(pos) == 3:
                    yi, xi = int(round(pos[-2])), int(round(pos[-1]))
                else:
                    yi, xi = int(round(pos[0])), int(round(pos[1]))
                photons = fluor.photons_per_frame * np.random.exponential(1)
                ph, pw = psf_2d.shape
                y0 = yi - ph // 2
                x0 = xi - pw // 2
                # Clip to image bounds
                sy = slice(max(0, y0), min(cfg.ny, y0 + ph))
                sx = slice(max(0, x0), min(cfg.nx, x0 + pw))
                py = slice(max(0, -y0), max(0, -y0) + sy.stop - sy.start)
                px = slice(max(0, -x0), max(0, -x0) + sx.stop - sx.start)
                frame[sy, sx] += psf_2d[py, px] * photons

            stack[t] = frame
        return stack

    def _render_dnapaint(self, structure, positions, psf, fluor,
                         _progress):
        """Render DNA-PAINT data."""
        cfg = self.config
        from .dynamics import dnapaint_binding_events

        if positions is None:
            positions = np.argwhere(structure > 0).astype(float)
            positions += np.random.uniform(-0.5, 0.5, positions.shape)

        n_sites = len(positions)
        if n_sites == 0:
            return np.zeros((cfg.nt, cfg.ny, cfg.nx))

        k_on = cfg.modality_params.get('k_on', 0.1)
        k_off = cfg.modality_params.get('k_off', 5.0)
        imager_conc = cfg.modality_params.get('imager_conc', 1.0)

        binding = dnapaint_binding_events(
            n_sites, cfg.nt, k_on, k_off, imager_conc, cfg.dt)

        stack = np.zeros((cfg.nt, cfg.ny, cfg.nx))
        psf_2d = psf if psf.ndim == 2 else psf[psf.shape[0] // 2]

        for t in range(cfg.nt):
            if t % max(1, cfg.nt // 20) == 0:
                pct = 40 + int(45 * t / max(cfg.nt, 1))
                _progress(pct, f"DNA-PAINT frame {t+1}/{cfg.nt}...")

            bound = binding[t]
            bound_pos = positions[bound]
            frame = np.zeros((cfg.ny, cfg.nx))
            for pos in bound_pos:
                yi = int(round(pos[-2] if len(pos) >= 2 else pos[0]))
                xi = int(round(pos[-1]))
                photons = fluor.photons_per_frame
                ph, pw = psf_2d.shape
                y0, x0 = yi - ph // 2, xi - pw // 2
                sy = slice(max(0, y0), min(cfg.ny, y0 + ph))
                sx = slice(max(0, x0), min(cfg.nx, x0 + pw))
                py = slice(max(0, -y0), max(0, -y0) + sy.stop - sy.start)
                px = slice(max(0, -x0), max(0, -x0) + sx.stop - sx.start)
                frame[sy, sx] += psf_2d[py, px] * photons
            stack[t] = frame
        return stack

    def _render_flim(self, structure, fluor, _progress):
        """Render FLIM data."""
        cfg = self.config
        from .dynamics import flim_image

        # Use 2D structure (single plane or max projection)
        if structure.ndim == 3:
            struct_2d = structure.max(axis=0)
        else:
            struct_2d = structure

        time_bins = cfg.modality_params.get('time_bins', 256)
        time_range = cfg.modality_params.get('time_range', 25.0)
        n_photons = cfg.modality_params.get('n_photons', 1000)

        _progress(50, "Generating FLIM histograms...")
        result = flim_image(struct_2d, fluor.lifetime, n_photons,
                            time_bins, time_range)
        # Return as (time_bins, H, W) for flika compatibility
        return result.transpose(2, 0, 1)

    def _positions_to_image(self, positions, shape):
        """Convert particle positions to an image."""
        image = np.zeros(shape)
        ndim = len(shape)
        for pos in positions:
            idx = tuple(int(round(p)) % s for p, s in
                        zip(pos[:ndim], shape))
            if all(0 <= idx[j] < shape[j] for j in range(ndim)):
                image[idx] = 1.0
        return image
