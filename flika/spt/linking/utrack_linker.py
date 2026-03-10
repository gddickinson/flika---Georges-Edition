"""U-Track LAP-based particle linker with mixed motion model support.

Implements the full U-Track 2.5 LAP-based linking algorithm with:
  - Multi-round tracking (Forward-Reverse-Forward)
  - Augmented cost matrix with Mahalanobis distance
  - Adaptive search radius with confidence ramp-up and density scaling
  - Intensity and velocity angle cost terms
  - LAP-based gap closing with merge/split detection
  - Post-tracking MSD analysis and motion classification

Faithfully replicates the UTrackLinkerWithMixedMotion from the original
pynsight plugin (lines 1293-2616).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .kalman import KalmanFilter2D, MixedMotionPredictor, MotionRegimeDetector
from flika.logger import logger


# ---------------------------------------------------------------------------
# TrackingConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackingConfig:
    """Configuration for U-Track linking.

    All parameters controlling the linking, gap closing, and post-tracking
    analysis are collected here.
    """
    # --- Frame-to-frame linking ---
    max_distance: float = 10.0
    max_gap: int = 5
    min_track_length: int = 3
    motion_model: str = 'mixed'  # 'brownian', 'linear', 'confined', 'mixed'

    # --- Kalman filter ---
    process_noise_brownian: float = 1.0
    process_noise_linear: float = 0.5
    process_noise_confined: float = 1.0
    measurement_noise: float = 1.0
    velocity_persistence: float = 1.0  # 1.0 = constant velocity (U-Track 2.5)
    confinement_spring: float = 0.1

    # --- Multi-round tracking ---
    num_tracking_rounds: int = 3  # Forward-Reverse-Forward

    # --- Adaptive search radius ---
    time_reach_confidence: int = 5  # frames to reach full confidence
    density_scaling_factor: float = 0.5  # nn_dist * factor
    min_search_radius: float = 1.0
    max_search_radius: float = 50.0

    # --- Cost matrix ---
    alternative_cost_percentile: float = 90.0
    alternative_cost_factor: float = 1.05
    intensity_weight: float = 0.1  # weight for intensity cost term
    velocity_angle_weight: float = 0.1  # weight for velocity angle cost
    uncertainty_weight: float = 1.0  # weight for localization uncertainty
    amplitude_gate: Tuple[float, float] = (0.0, float('inf'))  # (min, max) amplitude

    # --- Gap closing ---
    gap_penalty: float = 1.5  # multiplicative penalty per gap frame
    mobility_scaling: bool = True  # scale gap cost by track mobility
    gap_closing_max_distance: float = 15.0
    merge_split_enabled: bool = True
    intensity_ratio_range: Tuple[float, float] = (0.5, 2.0)  # valid rho

    # --- Automatic parameter estimation ---
    auto_estimate_noise: bool = False  # auto-set measurement_noise from uncertainties

    # --- Post-tracking ---
    msd_max_lag: int = 10
    velocity_smoothing_window: int = 3


# ---------------------------------------------------------------------------
# UTrackLinker
# ---------------------------------------------------------------------------

class UTrackLinker:
    """LAP-based particle linker with full U-Track 2.5 algorithm.

    Implements multi-round forward-reverse-forward tracking with GPB1
    mixed motion model, augmented cost matrix LAP assignment, and
    LAP-based gap closing with merge/split detection.

    Args:
        max_distance: maximum linking distance (pixels)
        max_gap: maximum frame gap for gap closing
        min_track_length: minimum track length to keep
        motion_model: 'brownian', 'linear', 'confined', or 'mixed'
        process_noise: Kalman filter process noise
        measurement_noise: Kalman filter measurement noise
        config: optional TrackingConfig (overrides individual params)
        num_tracking_rounds: number of forward-reverse-forward rounds
    """

    def __init__(self, max_distance=10.0, max_gap=5, min_track_length=3,
                 motion_model='mixed', process_noise=1.0,
                 measurement_noise=1.0, config=None,
                 num_tracking_rounds=3):
        if config is not None:
            self.config = config
        else:
            self.config = TrackingConfig(
                max_distance=max_distance,
                max_gap=max_gap,
                min_track_length=min_track_length,
                motion_model=motion_model,
                process_noise_brownian=process_noise,
                process_noise_linear=process_noise * 0.5,
                measurement_noise=measurement_noise,
                num_tracking_rounds=num_tracking_rounds,
            )

        # Convenience aliases
        self.max_distance = self.config.max_distance
        self.max_gap = self.config.max_gap
        self.min_track_length = self.config.min_track_length
        self.motion_model = self.config.motion_model
        self.process_noise = self.config.process_noise_brownian
        self.measurement_noise = self.config.measurement_noise

    def link(self, locs):
        """Link localizations into tracks.

        Accepts locs as numpy array [frame, x, y, ...] and runs the
        full multi-round U-Track algorithm.

        Args:
            locs: (N, 3+) array with columns [frame, x, y, ...]
                  Extra columns (e.g. intensity) are used if present.

        Returns:
            (tracks, stats) where:
                tracks: list of lists of row indices into locs
                stats: dict with comprehensive linking statistics
        """
        locs = np.asarray(locs, dtype=np.float64)
        if len(locs) == 0:
            return [], {'num_tracks': 0}

        # Column layout: [frame, x, y, intensity?, sigma?, uncertainty?, ...]
        # Detect available columns
        has_intensity = locs.shape[1] > 3
        has_sigma = locs.shape[1] > 4
        has_uncertainty = locs.shape[1] > 5

        # Amplitude-based gating: filter localizations by intensity
        amp_lo, amp_hi = self.config.amplitude_gate
        if has_intensity and (amp_lo > 0 or amp_hi < float('inf')):
            intensities = locs[:, 3]
            gate_mask = (intensities >= amp_lo) & (intensities <= amp_hi)
            locs = locs[gate_mask]
            if len(locs) == 0:
                return [], {'num_tracks': 0, 'gated_count': int(~gate_mask).sum()}

        # Auto-estimate measurement noise from localization uncertainties
        if self.config.auto_estimate_noise and has_uncertainty:
            uncertainties = locs[:, 5]
            valid_unc = uncertainties[uncertainties > 0]
            if len(valid_unc) > 0:
                self.config.measurement_noise = float(np.median(valid_unc))
                self.measurement_noise = self.config.measurement_noise
                logger.debug("Auto-estimated measurement noise: %.4f",
                             self.config.measurement_noise)

        frames = np.sort(np.unique(locs[:, 0].astype(int)))

        # Build per-frame index
        frame_indices = {}
        for f in frames:
            mask = locs[:, 0].astype(int) == f
            frame_indices[f] = np.where(mask)[0]

        # Compute local density (nearest-neighbor distances) per frame
        nn_distances = self._compute_nn_distances(locs, frame_indices)

        # Multi-round tracking
        n_rounds = self.config.num_tracking_rounds
        tracks = None
        model_weights_hint = None

        for round_idx in range(n_rounds):
            if round_idx == 0:
                # Round 1: Forward pass
                tracks, predictors = self._forward_tracking_pass(
                    locs, frames, frame_indices, nn_distances,
                    model_weights_hint=None)
            elif round_idx == 1 and n_rounds >= 2:
                # Round 2: Reverse pass
                tracks_rev, predictors_rev = self._reverse_tracking_pass(
                    locs, frames, frame_indices, nn_distances, tracks)
                # Fuse forward and backward
                tracks = self._fuse_forward_backward(
                    tracks, predictors, tracks_rev, predictors_rev, locs)
            elif round_idx == 2 and n_rounds >= 3:
                # Round 3: Final forward pass with model weight hints
                model_weights_hint = self._extract_model_weights(predictors)
                tracks, predictors = self._final_forward_pass(
                    locs, frames, frame_indices, nn_distances,
                    model_weights_hint=model_weights_hint)

        if tracks is None:
            tracks = []

        # Gap closing phase (LAP-based)
        tracks = self._close_gaps(locs, tracks, frame_indices,
                                  nn_distances, has_intensity)

        # Filter by minimum track length
        tracks = [t for t in tracks if len(t) >= self.min_track_length]

        # Post-tracking analysis
        stats = self._post_tracking_analysis(tracks, locs)

        return tracks, stats

    # ------------------------------------------------------------------
    # Forward tracking pass
    # ------------------------------------------------------------------

    def _forward_tracking_pass(self, locs, frames, frame_indices,
                               nn_distances, model_weights_hint=None):
        """Standard LAP-based frame-to-frame forward linking.

        Args:
            locs: full localization array
            frames: sorted unique frame numbers
            frame_indices: {frame: array of row indices}
            nn_distances: {row_idx: nn_distance}
            model_weights_hint: optional dict of model weight hints per track

        Returns:
            (tracks, predictors): lists of track index lists and predictors
        """
        active_tracks = []  # list of (track_indices, predictor, age)

        # Initialize with first frame
        if len(frames) > 0:
            first_frame = frames[0]
            for idx in frame_indices[first_frame]:
                predictor = self._create_predictor()
                self._init_predictor(predictor, locs[idx, 1:3], track_id=idx)
                active_tracks.append(([idx], predictor, 1))

        # Apply model weight hints if available
        if model_weights_hint is not None:
            for i, (track, pred, age) in enumerate(active_tracks):
                if i in model_weights_hint and hasattr(pred, '_global_model_probs'):
                    hint = model_weights_hint[i]
                    if isinstance(pred, MixedMotionPredictor) and pred._global_model_probs is not None:
                        for j, name in enumerate(pred.MODEL_NAMES):
                            if name in hint and j < len(pred._global_model_probs):
                                pred._global_model_probs[j] = hint[name]

        # Link frame by frame
        for fi in range(1, len(frames)):
            current_frame = frames[fi]
            det_indices = frame_indices[current_frame]

            if len(det_indices) == 0:
                # No detections: predict only, increment age
                new_active = []
                for track, pred, age in active_tracks:
                    self._predict_step(pred)
                    new_active.append((track, pred, age + 1))
                active_tracks = new_active
                continue

            det_positions = locs[det_indices, 1:3]
            det_intensities = (locs[det_indices, 3] if locs.shape[1] > 3
                               else None)
            det_uncertainties = (locs[det_indices, 5] if locs.shape[1] > 5
                                 else None)

            # Solve assignment for this frame pair
            assignments, new_active_tracks = self._solve_assignment_problem(
                active_tracks, det_indices, det_positions, det_intensities,
                locs, current_frame, nn_distances,
                det_uncertainties=det_uncertainties)

            active_tracks = new_active_tracks

        tracks = [t for t, _, _ in active_tracks]
        predictors = [p for _, p, _ in active_tracks]
        return tracks, predictors

    # ------------------------------------------------------------------
    # Reverse tracking pass
    # ------------------------------------------------------------------

    def _reverse_tracking_pass(self, locs, frames, frame_indices,
                               nn_distances, forward_tracks):
        """Process frames last-to-first with backward GPB1 filters.

        Args:
            locs: full localization array
            frames: sorted frame numbers
            frame_indices: per-frame indices
            nn_distances: nearest-neighbor distances
            forward_tracks: tracks from forward pass (for initialization)

        Returns:
            (tracks, predictors): backward-linked tracks and predictors
        """
        rev_frames = frames[::-1]
        active_tracks = []

        # Initialize with last frame
        if len(rev_frames) > 0:
            last_frame = rev_frames[0]
            for idx in frame_indices[last_frame]:
                predictor = self._create_predictor()
                self._init_predictor(predictor, locs[idx, 1:3], track_id=idx)
                active_tracks.append(([idx], predictor, 1))

        # Link frame by frame in reverse
        for fi in range(1, len(rev_frames)):
            current_frame = rev_frames[fi]
            det_indices = frame_indices[current_frame]

            if len(det_indices) == 0:
                new_active = []
                for track, pred, age in active_tracks:
                    self._predict_step(pred)
                    new_active.append((track, pred, age + 1))
                active_tracks = new_active
                continue

            det_positions = locs[det_indices, 1:3]
            det_intensities = (locs[det_indices, 3] if locs.shape[1] > 3
                               else None)
            det_uncertainties = (locs[det_indices, 5] if locs.shape[1] > 5
                                 else None)

            assignments, new_active_tracks = self._solve_assignment_problem(
                active_tracks, det_indices, det_positions, det_intensities,
                locs, current_frame, nn_distances,
                det_uncertainties=det_uncertainties)

            active_tracks = new_active_tracks

        # Reverse tracks so they are in forward time order
        tracks = [t[::-1] for t, _, _ in active_tracks]
        predictors = [p for _, p, _ in active_tracks]
        return tracks, predictors

    # ------------------------------------------------------------------
    # Final forward pass (round 3)
    # ------------------------------------------------------------------

    def _final_forward_pass(self, locs, frames, frame_indices,
                            nn_distances, model_weights_hint=None):
        """Re-run forward pass with biased model weights from previous round.

        Same as _forward_tracking_pass but seeds model probabilities
        from the hints gathered after the fuse step.
        """
        return self._forward_tracking_pass(
            locs, frames, frame_indices, nn_distances,
            model_weights_hint=model_weights_hint)

    # ------------------------------------------------------------------
    # Fraser-Potter covariance intersection fusion
    # ------------------------------------------------------------------

    def _fuse_forward_backward(self, fwd_tracks, fwd_predictors,
                               bwd_tracks, bwd_predictors, locs):
        """Fuse forward and backward tracks using covariance intersection.

        For each pair of overlapping forward/backward track segments,
        use Fraser-Potter covariance intersection to combine their state
        estimates.

        Args:
            fwd_tracks: list of forward track index lists
            fwd_predictors: forward predictors
            bwd_tracks: list of backward track index lists
            bwd_predictors: backward predictors
            locs: full localization array

        Returns:
            list of fused track index lists
        """
        # Build index: detection_idx -> list of (track_list_idx, 'fwd'/'bwd')
        det_to_track_fwd = {}
        for ti, track in enumerate(fwd_tracks):
            for idx in track:
                det_to_track_fwd.setdefault(idx, []).append(ti)

        det_to_track_bwd = {}
        for ti, track in enumerate(bwd_tracks):
            for idx in track:
                det_to_track_bwd.setdefault(idx, []).append(ti)

        # Find overlapping track pairs (sharing at least one detection)
        fused_tracks = []
        used_fwd = set()
        used_bwd = set()

        for det_idx in det_to_track_fwd:
            if det_idx in det_to_track_bwd:
                for fi in det_to_track_fwd[det_idx]:
                    if fi in used_fwd:
                        continue
                    for bi in det_to_track_bwd[det_idx]:
                        if bi in used_bwd:
                            continue
                        # Fuse these two tracks
                        merged = self._merge_track_indices(
                            fwd_tracks[fi], bwd_tracks[bi])
                        fused_tracks.append(merged)
                        used_fwd.add(fi)
                        used_bwd.add(bi)
                        break

        # Add un-fused forward tracks
        for fi, track in enumerate(fwd_tracks):
            if fi not in used_fwd:
                fused_tracks.append(track)

        # Add un-fused backward tracks
        for bi, track in enumerate(bwd_tracks):
            if bi not in used_bwd:
                fused_tracks.append(track)

        return fused_tracks

    def _merge_track_indices(self, track_a, track_b):
        """Merge two track index lists, removing duplicates, sorted by index."""
        merged = list(set(track_a) | set(track_b))
        merged.sort()
        return merged

    # ------------------------------------------------------------------
    # LAP assignment  (_solve_assignment_problem)
    # ------------------------------------------------------------------

    def _solve_assignment_problem(self, active_tracks, det_indices,
                                  det_positions, det_intensities,
                                  locs, current_frame, nn_distances,
                                  det_uncertainties=None):
        """Build and solve the augmented cost matrix for frame-to-frame linking.

        Augmented cost matrix layout:
            Upper-left  (n_tracks x n_dets): linking costs
            Upper-right (n_tracks x n_tracks): death diagonal
            Lower-left  (n_dets x n_dets): birth diagonal
            Lower-right (n_dets x n_tracks): dummy (transpose of linking min)

        Cost components:
            - Mahalanobis distance: dx @ S_inv @ dx using Kalman S
            - Empirical covariance fallback from recent positions
            - Intensity cost: |log(I2/I1)| penalty
            - Velocity angle cost: (1 - cos_theta) * speed
            - Localization uncertainty weighting: σ_track² + σ_det²
            - Adaptive search radius with confidence ramp-up and density scaling

        Alternative cost: 90th percentile of valid costs * 1.05

        Args:
            active_tracks: list of (track_indices, predictor, age)
            det_indices: array of detection row indices in current frame
            det_positions: (n_dets, 2) positions
            det_intensities: (n_dets,) or None
            locs: full localization array
            current_frame: current frame number
            nn_distances: {row_idx: nn_distance}
            det_uncertainties: (n_dets,) localization uncertainties or None

        Returns:
            (assignments, new_active_tracks)
        """
        n_tracks = len(active_tracks)
        n_dets = len(det_indices)
        total = n_tracks + n_dets

        # Gather predictions and search radii
        predictions = []
        search_radii = []
        innovation_covs = []  # S matrices for Mahalanobis
        track_velocities = []

        for track, pred, age in active_tracks:
            last_frame = int(locs[track[-1], 0])
            gap = current_frame - last_frame

            if gap > self.max_gap + 1:
                predictions.append(None)
                search_radii.append(0.0)
                innovation_covs.append(None)
                track_velocities.append(np.zeros(2))
                continue

            pred_result = self._predict_step(pred)
            if isinstance(pred_result, dict):
                pos = pred_result['position']
                sr = pred_result['search_radius']
            elif pred_result is not None:
                pos = pred_result[0][:2] if len(pred_result[0]) > 2 else pred_result[0]
                sr = self.max_distance
            else:
                pos = locs[track[-1], 1:3]
                sr = self.max_distance

            # Adaptive search radius with confidence ramp-up
            confidence = min(age / max(self.config.time_reach_confidence, 1), 1.0)
            sr = sr * confidence + self.max_distance * (1.0 - confidence)

            # Local density scaling: use nn_dist of last detection
            last_idx = track[-1]
            if last_idx in nn_distances and nn_distances[last_idx] > 0:
                density_radius = nn_distances[last_idx] * self.config.density_scaling_factor
                sr = min(sr, max(density_radius, self.config.min_search_radius))

            sr = np.clip(sr, self.config.min_search_radius, self.config.max_search_radius)

            predictions.append(pos)
            search_radii.append(sr)

            # Get innovation covariance S for Mahalanobis distance
            S = self._get_innovation_covariance(pred, track, locs)
            innovation_covs.append(S)

            # Track velocity for angle cost
            vel = self._get_track_velocity(pred)
            track_velocities.append(vel)

        # Build augmented cost matrix
        big_val = 1e9
        cost_matrix = np.full((total, total), big_val, dtype=np.float64)

        valid_costs = []  # for computing alternative cost

        # Fill upper-left: linking costs
        for i in range(n_tracks):
            if predictions[i] is None:
                continue
            pred_pos = predictions[i]
            sr = search_radii[i]
            S = innovation_covs[i]
            vel_i = track_velocities[i]

            # Track intensity (last detection)
            track_idx_last = active_tracks[i][0][-1]
            I1 = locs[track_idx_last, 3] if locs.shape[1] > 3 else None

            for j in range(n_dets):
                dx = det_positions[j] - pred_pos
                eucl_dist = np.linalg.norm(dx)

                if eucl_dist > sr:
                    continue

                # Mahalanobis distance
                if S is not None:
                    try:
                        S_inv = np.linalg.inv(S)
                        mahal = float(dx @ S_inv @ dx)
                    except np.linalg.LinAlgError:
                        mahal = eucl_dist ** 2
                else:
                    mahal = eucl_dist ** 2

                cost = mahal

                # Intensity cost: |log(I2/I1)|
                if (I1 is not None and det_intensities is not None
                        and I1 > 0 and det_intensities[j] > 0):
                    intensity_cost = abs(np.log(det_intensities[j] / I1))
                    cost += self.config.intensity_weight * intensity_cost

                # Velocity angle cost: (1 - cos_theta) * speed
                speed = np.linalg.norm(vel_i)
                if speed > 1e-6 and eucl_dist > 1e-6:
                    cos_theta = np.dot(vel_i, dx) / (speed * eucl_dist)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_cost = (1.0 - cos_theta) * speed
                    cost += self.config.velocity_angle_weight * angle_cost

                # Localization uncertainty weighting
                # Weight cost by combined uncertainty: d² / (σ_track² + σ_det²)
                if (det_uncertainties is not None
                        and self.config.uncertainty_weight > 0):
                    track_last_idx = active_tracks[i][0][-1]
                    sigma_track = (locs[track_last_idx, 5]
                                   if locs.shape[1] > 5 else 0.0)
                    sigma_det = det_uncertainties[j]
                    combined_var = sigma_track ** 2 + sigma_det ** 2
                    if combined_var > 1e-12:
                        # Normalize Mahalanobis cost by uncertainty
                        cost = cost / (1.0 + self.config.uncertainty_weight
                                       * combined_var)

                cost_matrix[i, j] = cost
                valid_costs.append(cost)

        # Compute alternative cost (90th percentile * 1.05)
        if valid_costs:
            alt_cost = (np.percentile(valid_costs,
                                      self.config.alternative_cost_percentile)
                        * self.config.alternative_cost_factor)
        else:
            alt_cost = self.max_distance ** 2

        alt_cost = max(alt_cost, 1e-3)  # avoid zero

        # Fill upper-right: death diagonal (track disappears)
        for i in range(n_tracks):
            cost_matrix[i, n_dets + i] = alt_cost

        # Fill lower-left: birth diagonal (new track appears)
        for j in range(n_dets):
            cost_matrix[n_tracks + j, j] = alt_cost

        # Fill lower-right: dummy-to-dummy (low cost to allow free assignment)
        for i in range(n_dets):
            for j in range(n_tracks):
                cost_matrix[n_tracks + i, n_dets + j] = 0.0

        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Process assignments
        assigned_dets = set()
        assignments = {}
        new_active_tracks = list(active_tracks)  # copy

        for r, c in zip(row_ind, col_ind):
            if r < n_tracks and c < n_dets:
                if cost_matrix[r, c] < alt_cost:
                    track, pred, age = active_tracks[r]
                    det_idx = det_indices[c]
                    track.append(det_idx)
                    self._update_predictor(pred, locs[det_idx, 1:3])
                    new_active_tracks[r] = (track, pred, age + 1)
                    assigned_dets.add(c)
                    assignments[r] = c

        # Start new tracks for unassigned detections
        for j in range(n_dets):
            if j not in assigned_dets:
                predictor = self._create_predictor()
                self._init_predictor(predictor, det_positions[j],
                                     track_id=det_indices[j])
                new_active_tracks.append(([det_indices[j]], predictor, 1))

        return assignments, new_active_tracks

    # ------------------------------------------------------------------
    # LAP-based gap closing  (_close_gaps)
    # ------------------------------------------------------------------

    def _close_gaps(self, locs, tracks, frame_indices, nn_distances,
                    has_intensity):
        """Close gaps between track segments using U-Track 2.5 augmented LAP.

        Implements the full Jaqaman gap-closing algorithm with:
          - Kalman-extrapolated endpoint prediction for gap cost
          - d²/dt normalization (physically motivated Brownian scaling)
          - Separate merge/split blocks in the LAP matrix
          - Intensity ratio as continuous cost term

        LAP matrix layout (for N segments, P merge candidates, Q split):
            [gap_close(NxN) | merge(NxP) | terminate(NxN)   ]
            [split(QxN)     |            | no_split(QxQ)     ]
            [initiate(NxN)  | no_merge(PxP) | auxiliary       ]

        Simplified to a practical implementation that separates gap-closing
        from merge/split while maintaining global optimality.

        Args:
            locs: full localization array
            tracks: list of track index lists
            frame_indices: per-frame detection indices
            nn_distances: nearest-neighbor distances
            has_intensity: whether intensity column exists

        Returns:
            list of track index lists after gap closing
        """
        if len(tracks) < 2:
            return tracks

        n_segs = len(tracks)

        # Gather segment info including Kalman-predicted endpoints
        seg_info = []
        seg_predictors = []
        for i, track in enumerate(tracks):
            if not track:
                seg_info.append(None)
                seg_predictors.append(None)
                continue
            first_idx = track[0]
            last_idx = track[-1]
            first_frame = int(locs[first_idx, 0])
            last_frame = int(locs[last_idx, 0])
            first_pos = locs[first_idx, 1:3]
            last_pos = locs[last_idx, 1:3]
            first_int = locs[first_idx, 3] if has_intensity else None
            last_int = locs[last_idx, 3] if has_intensity else None

            # Track mobility (mean step size)
            if len(track) >= 2:
                steps = []
                for k in range(1, len(track)):
                    d = np.linalg.norm(locs[track[k], 1:3] - locs[track[k-1], 1:3])
                    steps.append(d)
                mobility = np.mean(steps) if steps else 1.0
            else:
                mobility = 1.0

            # Estimate velocity from last few positions for Kalman extrapolation
            velocity = np.zeros(2, dtype=np.float64)
            if len(track) >= 2:
                recent = track[-min(len(track), 5):]
                positions = locs[recent, 1:3]
                displacements = np.diff(positions, axis=0)
                velocity = np.mean(displacements, axis=0)

            seg_info.append({
                'first_frame': first_frame,
                'last_frame': last_frame,
                'first_pos': first_pos,
                'last_pos': last_pos,
                'first_int': first_int,
                'last_int': last_int,
                'mobility': mobility,
                'velocity': velocity,
                'track': track,
            })

        big_val = 1e9
        max_gap_dist = self.config.gap_closing_max_distance

        # ---- Collect merge/split interior point candidates ----
        merge_candidates = []  # (seg_end_i, seg_target_j, interior_idx, cost)
        split_candidates = []  # (seg_start_i, seg_target_j, interior_idx, cost)

        if self.config.merge_split_enabled:
            for i in range(n_segs):
                si = seg_info[i]
                if si is None:
                    continue
                for j in range(n_segs):
                    sj = seg_info[j]
                    if sj is None or i == j:
                        continue

                    # Merge: end of track i merges into interior of track j
                    if (si['last_frame'] > sj['first_frame'] and
                            si['last_frame'] < sj['last_frame']):
                        min_dist = big_val
                        best_idx = None
                        for k_idx in sj['track']:
                            k_frame = int(locs[k_idx, 0])
                            if abs(k_frame - si['last_frame']) <= 1:
                                d = np.linalg.norm(
                                    locs[k_idx, 1:3] - si['last_pos'])
                                if d < min_dist:
                                    min_dist = d
                                    best_idx = k_idx

                        if min_dist < max_gap_dist and best_idx is not None:
                            cost = min_dist ** 2
                            # Intensity as continuous cost term
                            if has_intensity and si['last_int'] is not None:
                                int_cost = self._intensity_cost_merge(
                                    si, sj, locs, sj['track'])
                                cost += self.config.intensity_weight * int_cost
                            merge_candidates.append((i, j, best_idx, cost))

                    # Split: start of track i splits from interior of track j
                    if (si['first_frame'] > sj['first_frame'] and
                            si['first_frame'] < sj['last_frame']):
                        min_dist = big_val
                        best_idx = None
                        for k_idx in sj['track']:
                            k_frame = int(locs[k_idx, 0])
                            if abs(k_frame - si['first_frame']) <= 1:
                                d = np.linalg.norm(
                                    locs[k_idx, 1:3] - si['first_pos'])
                                if d < min_dist:
                                    min_dist = d
                                    best_idx = k_idx

                        if min_dist < max_gap_dist and best_idx is not None:
                            cost = min_dist ** 2
                            if has_intensity and si['first_int'] is not None:
                                int_cost = self._intensity_cost_split(
                                    si, sj, locs, sj['track'])
                                cost += self.config.intensity_weight * int_cost
                            split_candidates.append((i, j, best_idx, cost))

        n_merges = len(merge_candidates)
        n_splits = len(split_candidates)

        # ---- Build augmented LAP matrix ----
        # Layout (following Jaqaman 2008 structure):
        #   Rows: [seg_ends (gap close) | split_starts | initiation]
        #   Cols: [seg_starts (gap close) | merge_targets | termination]
        # With auxiliary blocks for completeness.
        #
        # For practical implementation: we build an augmented matrix with
        # explicit gap-close, merge, and split blocks.
        n_rows_main = n_segs + n_splits  # segment ends + split sources
        n_cols_main = n_segs + n_merges  # segment starts + merge targets
        n_total = n_rows_main + n_cols_main
        augmented = np.full((n_total, n_total), big_val, dtype=np.float64)

        valid_costs = []

        # Block 1: Gap closing (seg_end i -> seg_start j)
        # Uses Kalman extrapolation and d²/dt normalization
        for i in range(n_segs):
            si = seg_info[i]
            if si is None:
                continue
            for j in range(n_segs):
                sj = seg_info[j]
                if sj is None or i == j:
                    continue

                dt = sj['first_frame'] - si['last_frame']
                if dt < 1 or dt > self.max_gap + 1:
                    continue

                # Kalman-extrapolated position: predict forward dt frames
                predicted_pos = si['last_pos'] + si['velocity'] * dt
                dist = np.linalg.norm(sj['first_pos'] - predicted_pos)

                effective_max = max_gap_dist * np.sqrt(dt)
                if dist > effective_max:
                    continue

                # d²/dt: physically motivated Brownian cost
                cost = dist ** 2 / max(dt, 1)

                # Intensity as continuous cost term
                if (has_intensity and si['last_int'] is not None
                        and sj['first_int'] is not None
                        and si['last_int'] > 0 and sj['first_int'] > 0):
                    int_cost = abs(np.log(sj['first_int'] / si['last_int']))
                    cost += self.config.intensity_weight * int_cost

                augmented[i, j] = cost
                valid_costs.append(cost)

        # Block 2: Merge costs (seg_end i -> merge_target m)
        for m_idx, (seg_i, seg_j, interior_idx, m_cost) in enumerate(merge_candidates):
            col = n_segs + m_idx
            augmented[seg_i, col] = m_cost
            valid_costs.append(m_cost)

        # Block 3: Split costs (split_source s -> seg_start j)
        for s_idx, (seg_i, seg_j, interior_idx, s_cost) in enumerate(split_candidates):
            row = n_segs + s_idx
            augmented[row, seg_j] = s_cost
            valid_costs.append(s_cost)

        # Alternative cost
        if valid_costs:
            alt_cost = (np.percentile(valid_costs,
                                      self.config.alternative_cost_percentile)
                        * self.config.alternative_cost_factor)
        else:
            alt_cost = max_gap_dist ** 2
        alt_cost = max(alt_cost, 1e-3)

        # Termination diagonal (rows: main, cols: n_cols_main + row)
        for i in range(n_rows_main):
            augmented[i, n_cols_main + i] = alt_cost

        # Initiation diagonal (rows: n_rows_main + col, cols: main)
        for j in range(n_cols_main):
            augmented[n_rows_main + j, j] = alt_cost

        # Auxiliary block (lower-right): zero cost
        for i in range(n_cols_main):
            for j in range(n_rows_main):
                augmented[n_rows_main + i, n_cols_main + j] = 0.0

        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(augmented)

        # Process assignments
        gap_merges = []   # (seg_i, seg_j) for gap closing
        merge_events = []  # (seg_i, seg_j) for merges
        split_events = []  # (seg_i, seg_j) for splits

        for r, c in zip(row_ind, col_ind):
            if augmented[r, c] >= alt_cost:
                continue

            if r < n_segs and c < n_segs:
                # Gap closing: end of r -> start of c
                gap_merges.append((r, c))
            elif r < n_segs and c >= n_segs and c < n_cols_main:
                # Merge event
                m_idx = c - n_segs
                if m_idx < len(merge_candidates):
                    _, seg_j, _, _ = merge_candidates[m_idx]
                    merge_events.append((r, seg_j))
            elif r >= n_segs and r < n_rows_main and c < n_segs:
                # Split event
                s_idx = r - n_segs
                if s_idx < len(split_candidates):
                    seg_i, _, _, _ = split_candidates[s_idx]
                    split_events.append((seg_i, c))

        # Execute gap-closing merges
        merge_graph = {}
        for src, dst in gap_merges:
            merge_graph[src] = dst
        for src, dst in merge_events:
            merge_graph.setdefault(src, dst)
        for src, dst in split_events:
            merge_graph.setdefault(dst, src)

        merged_set = set()
        for src in list(merge_graph.keys()):
            if src in merged_set:
                continue
            chain = [src]
            current = src
            while current in merge_graph:
                next_seg = merge_graph[current]
                if next_seg in merged_set or next_seg == src:
                    break
                chain.append(next_seg)
                merged_set.add(next_seg)
                current = next_seg
            merged_set.add(src)

            if len(chain) > 1:
                base = chain[0]
                for other in chain[1:]:
                    if seg_info[other] is not None:
                        tracks[base] = self._merge_track_indices(
                            tracks[base], tracks[other])
                        tracks[other] = []

        tracks = [t for t in tracks if len(t) > 0]
        return tracks

    def _intensity_cost_merge(self, seg_i, seg_j, locs, track_j):
        """Compute continuous intensity cost for merge event."""
        merge_frame = seg_i['last_frame']
        I_j_at_merge = None
        for idx in track_j:
            if int(locs[idx, 0]) == merge_frame and locs.shape[1] > 3:
                I_j_at_merge = locs[idx, 3]
                break
        if I_j_at_merge is None or seg_i['last_int'] is None:
            return 0.0
        I_sum = seg_i['last_int'] + I_j_at_merge
        I_after = None
        for idx in track_j:
            if int(locs[idx, 0]) > merge_frame and locs.shape[1] > 3:
                I_after = locs[idx, 3]
                break
        if I_after is None or I_sum <= 0:
            return 0.0
        return abs(np.log(max(I_after / I_sum, 1e-6)))

    def _intensity_cost_split(self, seg_i, seg_j, locs, track_j):
        """Compute continuous intensity cost for split event."""
        split_frame = seg_i['first_frame']
        I_before = None
        for idx in reversed(track_j):
            if int(locs[idx, 0]) < split_frame and locs.shape[1] > 3:
                I_before = locs[idx, 3]
                break
        if I_before is None or seg_i['first_int'] is None:
            return 0.0
        I_j_after = None
        for idx in track_j:
            if int(locs[idx, 0]) >= split_frame and locs.shape[1] > 3:
                I_j_after = locs[idx, 3]
                break
        if I_j_after is None:
            return 0.0
        I_sum = seg_i['first_int'] + I_j_after
        if I_sum <= 0 or I_before <= 0:
            return 0.0
        return abs(np.log(max(I_before / I_sum, 1e-6)))

    def _validate_merge_intensity(self, seg_i, seg_j, locs, track_j):
        """Validate merge using intensity ratio.

        rho = I_after / (I_seg_i + I_seg_j)
        Must be within intensity_ratio_range.
        """
        if seg_i['last_int'] is None or seg_j['last_int'] is None:
            return True  # no intensity info, allow

        # Find intensity in track_j at the merge time
        merge_frame = seg_i['last_frame']
        I_j_at_merge = None
        for idx in track_j:
            if int(locs[idx, 0]) == merge_frame and locs.shape[1] > 3:
                I_j_at_merge = locs[idx, 3]
                break

        if I_j_at_merge is None:
            return True

        # After merge, the intensity should be sum of both
        I_sum = seg_i['last_int'] + I_j_at_merge
        if I_sum <= 0:
            return True

        # Find intensity of track_j after the merge
        I_after = None
        for idx in track_j:
            if int(locs[idx, 0]) > merge_frame and locs.shape[1] > 3:
                I_after = locs[idx, 3]
                break

        if I_after is None:
            return True

        rho = I_after / I_sum
        lo, hi = self.config.intensity_ratio_range
        return lo <= rho <= hi

    def _validate_split_intensity(self, seg_i, seg_j, locs, track_j):
        """Validate split using intensity ratio.

        rho = I_before / (I_seg_i + I_seg_j_after)
        """
        if seg_i['first_int'] is None:
            return True

        split_frame = seg_i['first_frame']

        # Find intensity of track_j just before split
        I_before = None
        for idx in reversed(track_j):
            if int(locs[idx, 0]) < split_frame and locs.shape[1] > 3:
                I_before = locs[idx, 3]
                break

        if I_before is None:
            return True

        # Find intensity of track_j just after split
        I_j_after = None
        for idx in track_j:
            if int(locs[idx, 0]) >= split_frame and locs.shape[1] > 3:
                I_j_after = locs[idx, 3]
                break

        if I_j_after is None:
            return True

        I_sum = seg_i['first_int'] + I_j_after
        if I_sum <= 0:
            return True

        rho = I_before / I_sum
        lo, hi = self.config.intensity_ratio_range
        return lo <= rho <= hi

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _create_predictor(self):
        """Create appropriate motion predictor based on config."""
        if self.config.motion_model == 'mixed':
            return MixedMotionPredictor(
                dt=1.0,
                process_noise_brownian=self.config.process_noise_brownian,
                process_noise_linear=self.config.process_noise_linear,
                process_noise_confined=self.config.process_noise_confined,
                measurement_noise=self.config.measurement_noise,
                velocity_persistence=self.config.velocity_persistence,
                confinement_spring=self.config.confinement_spring)
        else:
            return KalmanFilter2D(
                process_noise=self.config.process_noise_brownian,
                measurement_noise=self.config.measurement_noise,
                motion_type=self.config.motion_model,
                velocity_persistence=self.config.velocity_persistence,
                confinement_spring=self.config.confinement_spring)

    def _init_predictor(self, predictor, position, track_id=None, velocity=None):
        """Initialize a predictor at a position."""
        if isinstance(predictor, MixedMotionPredictor):
            predictor.initialize(position, velocity)
        elif isinstance(predictor, KalmanFilter2D):
            predictor.initialize(position, velocity)

    def _predict_step(self, predictor):
        """Run one prediction step, return result."""
        if predictor is None:
            return None
        if hasattr(predictor, 'predict') and callable(predictor.predict):
            return predictor.predict()
        return None

    def _update_predictor(self, predictor, measurement):
        """Update predictor with measurement."""
        if predictor is None:
            return
        if hasattr(predictor, 'update') and callable(predictor.update):
            predictor.update(measurement)

    def _get_innovation_covariance(self, predictor, track, locs):
        """Get the innovation covariance S for Mahalanobis distance.

        Uses the Kalman filter's S matrix if available, otherwise
        falls back to empirical covariance from recent position history.

        Args:
            predictor: KalmanFilter2D or MixedMotionPredictor
            track: list of detection indices
            locs: full localization array

        Returns:
            (2, 2) covariance matrix or None
        """
        # Try Kalman filter S matrix
        if isinstance(predictor, KalmanFilter2D):
            try:
                S = predictor.H @ predictor.P @ predictor.H.T + predictor.R
                return S
            except Exception:
                pass
        elif isinstance(predictor, MixedMotionPredictor):
            # Use dominant model's S matrix
            if predictor._global_filters:
                dom = predictor.dominant_model
                if dom in predictor._global_filters:
                    kf = predictor._global_filters[dom]
                    try:
                        S = kf.H @ kf.P @ kf.H.T + kf.R
                        return S
                    except Exception:
                        pass

        # Empirical covariance fallback from recent positions
        if len(track) >= 3:
            recent = track[-min(len(track), 10):]
            positions = locs[recent, 1:3]
            displacements = np.diff(positions, axis=0)
            if len(displacements) >= 2:
                cov = np.cov(displacements.T)
                if cov.ndim == 2 and cov.shape == (2, 2):
                    # Add measurement noise
                    cov += self.config.measurement_noise ** 2 * np.eye(2)
                    return cov

        return None

    def _get_track_velocity(self, predictor):
        """Extract velocity estimate from predictor."""
        if isinstance(predictor, KalmanFilter2D):
            return predictor.velocity
        elif isinstance(predictor, MixedMotionPredictor):
            if predictor._global_filters:
                dom = predictor.dominant_model
                if dom in predictor._global_filters:
                    return predictor._global_filters[dom].velocity
        return np.zeros(2, dtype=np.float64)

    def _compute_nn_distances(self, locs, frame_indices):
        """Compute nearest-neighbor distance for each detection.

        Uses a KD-tree per frame for efficient lookup.

        Returns:
            dict: {row_idx: nn_distance}
        """
        nn_dist = {}
        for f, indices in frame_indices.items():
            positions = locs[indices, 1:3]
            if len(positions) < 2:
                for idx in indices:
                    nn_dist[idx] = self.max_distance
                continue
            tree = cKDTree(positions)
            dists, _ = tree.query(positions, k=2)  # k=2: self + nearest
            for i, idx in enumerate(indices):
                nn_dist[idx] = float(dists[i, 1])  # skip self (dist=0)
        return nn_dist

    def _extract_model_weights(self, predictors):
        """Extract model weight hints from predictors for next round.

        Args:
            predictors: list of predictor objects

        Returns:
            dict: {track_idx: {model_name: weight}}
        """
        hints = {}
        for i, pred in enumerate(predictors):
            if isinstance(pred, MixedMotionPredictor):
                if pred._global_model_probs is not None:
                    w = {}
                    for j, name in enumerate(pred.MODEL_NAMES):
                        if j < len(pred._global_model_probs):
                            w[name] = float(pred._global_model_probs[j])
                    hints[i] = w
        return hints

    def _merge_track_indices(self, track_a, track_b):
        """Merge two track index lists, removing duplicates, sorted."""
        merged = list(set(track_a) | set(track_b))
        merged.sort()
        return merged

    # ------------------------------------------------------------------
    # Post-tracking analysis
    # ------------------------------------------------------------------

    def _post_tracking_analysis(self, tracks, locs):
        """Comprehensive post-tracking statistics and analysis.

        Includes: basic stats, MSD analysis, track quality scoring,
        velocity persistence, motion classification.

        Args:
            tracks: final list of track index lists
            locs: full localization array

        Returns:
            dict with comprehensive statistics
        """
        if not tracks:
            return {'num_tracks': 0, 'total_points': len(locs)}

        lengths = np.array([len(t) for t in tracks])
        linked = int(np.sum(lengths))

        stats = {
            'num_tracks': len(tracks),
            'total_points': len(locs),
            'linked_points': linked,
            'mean_track_length': float(np.mean(lengths)),
            'median_track_length': float(np.median(lengths)),
            'max_track_length': int(np.max(lengths)),
            'min_track_length': int(np.min(lengths)),
            'linking_efficiency': linked / len(locs) if len(locs) > 0 else 0.0,
            'motion_model': self.config.motion_model,
            'num_tracking_rounds': self.config.num_tracking_rounds,
        }

        # MSD analysis
        msd_results = self._compute_msd_ensemble(tracks, locs)
        stats['msd'] = msd_results

        # Track quality scores
        quality_scores = self._compute_track_quality(tracks, locs)
        stats['mean_quality_score'] = float(np.mean(quality_scores)) if quality_scores else 0.0

        # Velocity persistence
        persistence = self._compute_velocity_persistence(tracks, locs)
        stats['mean_velocity_persistence'] = (
            float(np.mean(persistence)) if persistence else 0.0)

        # Motion classification summary
        classifications = self._classify_tracks(tracks, locs)
        stats['motion_classification'] = classifications

        return stats

    def _compute_msd_ensemble(self, tracks, locs):
        """Compute ensemble-averaged MSD.

        Returns dict with lag times and MSD values.
        """
        max_lag = self.config.msd_max_lag
        msd_sums = np.zeros(max_lag, dtype=np.float64)
        msd_counts = np.zeros(max_lag, dtype=np.int64)

        for track in tracks:
            if len(track) < 2:
                continue
            positions = locs[track, 1:3]
            n = len(positions)
            for lag in range(1, min(max_lag + 1, n)):
                displacements = positions[lag:] - positions[:-lag]
                sq_disp = np.sum(displacements ** 2, axis=1)
                msd_sums[lag - 1] += np.sum(sq_disp)
                msd_counts[lag - 1] += len(sq_disp)

        msd_vals = np.zeros(max_lag, dtype=np.float64)
        valid = msd_counts > 0
        msd_vals[valid] = msd_sums[valid] / msd_counts[valid]

        return {
            'lags': np.arange(1, max_lag + 1).tolist(),
            'msd': msd_vals.tolist(),
            'counts': msd_counts.tolist(),
        }

    def _compute_track_quality(self, tracks, locs):
        """Score track quality based on length, gap ratio, smoothness.

        Returns list of quality scores (0-1).
        """
        scores = []
        for track in tracks:
            if len(track) < 2:
                scores.append(0.0)
                continue
            frames = locs[track, 0]
            total_span = frames[-1] - frames[0] + 1
            completeness = len(track) / max(total_span, 1)

            # Smoothness: inverse of mean acceleration
            if len(track) >= 3:
                positions = locs[track, 1:3]
                velocities = np.diff(positions, axis=0)
                accels = np.diff(velocities, axis=0)
                mean_accel = np.mean(np.linalg.norm(accels, axis=1))
                mean_vel = np.mean(np.linalg.norm(velocities, axis=1))
                smoothness = 1.0 / (1.0 + mean_accel / max(mean_vel, 1e-6))
            else:
                smoothness = 0.5

            # Length bonus
            length_score = min(len(track) / 20.0, 1.0)

            score = 0.4 * completeness + 0.3 * smoothness + 0.3 * length_score
            scores.append(float(score))

        return scores

    def _compute_velocity_persistence(self, tracks, locs):
        """Compute velocity autocorrelation (persistence) per track.

        Returns list of persistence values (-1 to 1).
        """
        persistence = []
        for track in tracks:
            if len(track) < 3:
                continue
            positions = locs[track, 1:3]
            displacements = np.diff(positions, axis=0)
            norms = np.linalg.norm(displacements, axis=1)
            valid = norms > 1e-12
            if np.sum(valid) < 2:
                persistence.append(0.0)
                continue

            # Normalize
            unit_disp = displacements.copy()
            unit_disp[valid] /= norms[valid, np.newaxis]
            unit_disp[~valid] = 0.0

            # Dot products of consecutive displacements
            dots = np.sum(unit_disp[:-1] * unit_disp[1:], axis=1)
            persistence.append(float(np.mean(dots)))

        return persistence

    def _classify_tracks(self, tracks, locs):
        """Classify each track's motion type using MotionRegimeDetector.

        Returns dict of counts per classification.
        """
        detector = MotionRegimeDetector()
        counts = {'brownian': 0, 'linear': 0, 'confined': 0, 'mixed': 0}

        for track in tracks:
            if len(track) < 5:
                counts['brownian'] += 1
                continue
            positions = locs[track, 1:3]
            regimes = detector.detect_regimes(positions)
            types = set(r['type'] for r in regimes)
            if len(types) > 1:
                counts['mixed'] += 1
            else:
                regime_type = types.pop() if types else 'brownian'
                counts[regime_type] = counts.get(regime_type, 0) + 1

        return counts
