"""Kalman filter and motion regime detection for particle tracking.

Provides a 2D Kalman filter supporting Brownian, linear, and confined motion
models, a sliding-window motion regime detector, and a GPB1 (Generalized
Pseudo-Bayesian order 1) mixed motion predictor with per-track filter banks
for heterogeneous tracking.

Faithfully replicates the algorithms from the original pynsight plugin
(lines 687-1292).
"""
import copy
import numpy as np
from ...logger import logger


# ---------------------------------------------------------------------------
# KalmanFilter2D  (plugin lines 802-953)
# ---------------------------------------------------------------------------

class KalmanFilter2D:
    """2D Kalman filter with position + velocity state.

    State vector: [x, y, vx, vy]  (4D)
    Observation:  [x, y]           (2D)

    Supports three motion types:
      - 'brownian':  velocity decays to zero each step (F[2,2]=0, F[3,3]=0)
      - 'linear':    velocity persists with configurable decay
                     (F[2,2]=velocity_persistence, F[3,3]=velocity_persistence)
      - 'confined':  Ornstein-Uhlenbeck restoring force toward a
                     confinement_center; control input applied in predict()

    Process noise Q is diagonal: [q*dt, q*dt, q, q].
    Covariance update uses the numerically stable Joseph form.
    Innovation likelihood uses slogdet for numerical stability.

    Args:
        dt: time step between frames
        process_noise: process noise magnitude q
        measurement_noise: measurement noise magnitude
        motion_type: one of 'brownian', 'linear', 'confined'
        velocity_persistence: fraction of velocity retained per step
            (only used for 'linear' motion_type, default 0.8)
        confinement_spring: spring constant k for confined motion
            (only used for 'confined' motion_type, default 0.1)
        confinement_center: (x, y) center of confinement region
            (only used for 'confined' motion_type)
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=1.0,
                 motion_type='brownian', velocity_persistence=0.8,
                 confinement_spring=0.1, confinement_center=None):
        self.dt = dt
        self.motion_type = motion_type
        self.velocity_persistence = velocity_persistence
        self.confinement_spring = confinement_spring
        self.confinement_center = (np.array(confinement_center, dtype=np.float64)
                                   if confinement_center is not None
                                   else np.zeros(2, dtype=np.float64))

        q = process_noise ** 2

        # --- State transition matrix F ---
        # Base: position += velocity * dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.float64)

        if motion_type == 'brownian':
            # Velocity decays to zero: F[2,2]=0, F[3,3]=0  (already set)
            pass
        elif motion_type == 'linear':
            # Velocity persists with decay factor
            self.F[2, 2] = velocity_persistence
            self.F[3, 3] = velocity_persistence
        elif motion_type == 'confined':
            # Ornstein-Uhlenbeck: position pulled toward center
            # F[0,0] = 1 - k*dt, F[1,1] = 1 - k*dt
            k = confinement_spring
            self.F[0, 0] = 1.0 - k * dt
            self.F[1, 1] = 1.0 - k * dt
            # Velocity still decays (no persistence for confined)
            self.F[2, 2] = 0.0
            self.F[3, 3] = 0.0
        else:
            raise ValueError(f"Unknown motion_type: {motion_type!r}. "
                             f"Must be 'brownian', 'linear', or 'confined'.")

        # --- Observation matrix H ---
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # --- Process noise Q: diagonal [q*dt, q*dt, q, q] ---
        self.Q = np.diag([q * dt, q * dt, q, q]).astype(np.float64)

        # --- Measurement noise R ---
        self.R = measurement_noise ** 2 * np.eye(2, dtype=np.float64)

        # --- State and covariance ---
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0  # large initial uncertainty

    def initialize(self, position, velocity=None):
        """Initialize filter state.

        Args:
            position: (2,) array [x, y]
            velocity: optional (2,) array [vx, vy]
        """
        self.x[:2] = np.asarray(position, dtype=np.float64)
        if velocity is not None:
            self.x[2:] = np.asarray(velocity, dtype=np.float64)
        else:
            self.x[2:] = 0.0
        self.P = np.eye(4, dtype=np.float64) * 100.0

    def predict(self, confinement_center=None):
        """Predict next state.

        For confined motion, a control input is added to pull the state
        toward the confinement center.

        Args:
            confinement_center: optional (x, y) override for confined motion

        Returns:
            (predicted_state, predicted_covariance) tuple
        """
        self.x = self.F @ self.x

        # Confined motion: add restoring control input toward center
        if self.motion_type == 'confined':
            center = (np.asarray(confinement_center, dtype=np.float64)
                      if confinement_center is not None
                      else self.confinement_center)
            k = self.confinement_spring
            dt = self.dt
            # Control input: restoring force pulls position toward center
            self.x[0] += k * dt * center[0]
            self.x[1] += k * dt * center[1]

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, measurement):
        """Update state with measurement using Joseph-form covariance update.

        Joseph form: P = (I - K*H) @ P_pred @ (I - K*H).T + K @ R @ K.T
        This is numerically more stable than the standard form.

        Args:
            measurement: (2,) array [x, y]

        Returns:
            (updated_state, updated_covariance) tuple
        """
        measurement = np.asarray(measurement, dtype=np.float64)

        # Innovation
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ self.H.T @ S_inv

        # State update
        self.x = self.x + K @ y

        # Joseph-form covariance update (numerically stable)
        I = np.eye(4, dtype=np.float64)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    def innovation_likelihood(self, measurement):
        """Compute Gaussian innovation likelihood N(y; 0, S).

        Uses slogdet for numerical stability instead of computing the
        determinant directly.

        Args:
            measurement: (2,) array [x, y]

        Returns:
            likelihood: float, probability density of the innovation
        """
        measurement = np.asarray(measurement, dtype=np.float64)
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Use slogdet for numerical stability
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e-300  # degenerate covariance

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 1e-300

        d = len(y)  # dimension = 2
        mahal = y @ S_inv @ y
        log_likelihood = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)

        # Clamp to avoid underflow
        log_likelihood = max(log_likelihood, -700.0)
        return np.exp(log_likelihood)

    def get_search_radius(self, sigma_factor=3.0):
        """Adaptive search radius from eigenvalues of innovation covariance S.

        The search radius is sigma_factor times the square root of the
        largest eigenvalue of the innovation covariance matrix S = H*P*H' + R.

        Args:
            sigma_factor: number of standard deviations (default 3.0)

        Returns:
            radius: float, adaptive search radius in pixels
        """
        S = self.H @ self.P @ self.H.T + self.R
        eigvals = np.linalg.eigvalsh(S)
        # Use largest eigenvalue for conservative radius
        radius = sigma_factor * np.sqrt(np.max(np.abs(eigvals)))
        return float(radius)

    def copy(self):
        """Deep copy of this filter (for filter bank operations)."""
        return copy.deepcopy(self)

    @property
    def position(self):
        """Current position estimate [x, y]."""
        return self.x[:2].copy()

    @property
    def velocity(self):
        """Current velocity estimate [vx, vy]."""
        return self.x[2:].copy()

    @property
    def state(self):
        """Full state vector [x, y, vx, vy]."""
        return self.x.copy()


# ---------------------------------------------------------------------------
# MotionRegimeDetector  (plugin lines 687-800)
# ---------------------------------------------------------------------------

class MotionRegimeDetector:
    """Sliding-window motion regime detector.

    Classifies track segments into three regimes:
      - 'brownian': random walk, low directional persistence
      - 'linear':   directed / persistent motion
      - 'confined': spatially restricted motion

    Uses directional persistence analysis: computes dot products of
    consecutive normalized displacement vectors within a sliding window.

    Short regime segments are merged into neighboring regimes
    (_merge_short_regimes).

    Args:
        window_size: sliding window size for local analysis
        persistence_threshold: cosine threshold for linear detection (0-1)
        confinement_threshold: Rg/step_size ratio threshold for confined
        min_regime_length: minimum segment length; shorter segments are merged
    """

    def __init__(self, window_size=11, persistence_threshold=0.5,
                 confinement_threshold=0.5, min_regime_length=5):
        self.window_size = window_size
        self.persistence_threshold = persistence_threshold
        self.confinement_threshold = confinement_threshold
        self.min_regime_length = min_regime_length

    def detect_regimes(self, positions):
        """Classify motion regime for each point in a track.

        Args:
            positions: (N, 2) array of [x, y] coordinates

        Returns:
            list of dicts, each with keys:
                'start': int, start index (inclusive)
                'end': int, end index (exclusive)
                'type': str, one of 'brownian', 'linear', 'confined'
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)

        if n < 3:
            return [{'start': 0, 'end': n, 'type': 'brownian'}]

        # Compute displacement vectors
        displacements = np.diff(positions, axis=0)  # (N-1, 2)
        norms = np.linalg.norm(displacements, axis=1)  # (N-1,)

        # Normalize displacement vectors (handle zero-length)
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        unit_displacements = displacements / safe_norms[:, np.newaxis]

        # Compute per-point regime labels using sliding window
        per_point_labels = ['brownian'] * n
        half_win = self.window_size // 2

        for i in range(n):
            start = max(0, i - half_win)
            end = min(n, i + half_win + 1)
            seg = positions[start:end]

            if len(seg) < 3:
                continue

            # --- Directional persistence via dot products ---
            # Get displacement vectors in the window
            d_start = max(0, start)
            d_end = min(n - 1, end - 1)
            if d_end - d_start < 2:
                continue

            local_units = unit_displacements[d_start:d_end]
            if len(local_units) < 2:
                continue

            # Dot product of consecutive normalized displacement vectors
            dots = np.sum(local_units[:-1] * local_units[1:], axis=1)
            mean_persistence = np.mean(dots)

            # Check for linear motion (high directional persistence)
            if mean_persistence > self.persistence_threshold:
                per_point_labels[i] = 'linear'
                continue

            # Check for confined motion (low Rg relative to step size)
            center = seg.mean(axis=0)
            rg = np.sqrt(np.mean(np.sum((seg - center) ** 2, axis=1)))
            local_norms = norms[d_start:d_end]
            mean_step = np.mean(local_norms) if len(local_norms) > 0 else 0.0

            if mean_step > 1e-12 and rg / mean_step < self.confinement_threshold:
                per_point_labels[i] = 'confined'

        # Convert per-point labels to regime segments
        regimes = self._labels_to_segments(per_point_labels)

        # Merge short regimes into neighbors
        regimes = self._merge_short_regimes(regimes, n)

        return regimes

    def _labels_to_segments(self, labels):
        """Convert per-point labels into contiguous regime segments.

        Args:
            labels: list of str, one per point

        Returns:
            list of dicts with 'start', 'end', 'type'
        """
        if not labels:
            return []

        segments = []
        current_type = labels[0]
        current_start = 0

        for i in range(1, len(labels)):
            if labels[i] != current_type:
                segments.append({
                    'start': current_start,
                    'end': i,
                    'type': current_type
                })
                current_type = labels[i]
                current_start = i

        # Final segment
        segments.append({
            'start': current_start,
            'end': len(labels),
            'type': current_type
        })

        return segments

    def _merge_short_regimes(self, regimes, n):
        """Merge regime segments shorter than min_regime_length.

        Short segments are absorbed into the neighboring segment
        (preferring the longer neighbor, or the preceding one on ties).

        Args:
            regimes: list of regime dicts
            n: total number of points

        Returns:
            list of regime dicts after merging
        """
        if len(regimes) <= 1:
            return regimes

        changed = True
        while changed:
            changed = False
            merged = []
            i = 0
            while i < len(regimes):
                seg = regimes[i]
                seg_len = seg['end'] - seg['start']

                if seg_len < self.min_regime_length and len(regimes) > 1:
                    # Determine which neighbor to merge into
                    if i == 0:
                        # Merge into next
                        if i + 1 < len(regimes):
                            regimes[i + 1]['start'] = seg['start']
                            changed = True
                            i += 1
                            continue
                    elif i == len(regimes) - 1:
                        # Merge into previous
                        if merged:
                            merged[-1]['end'] = seg['end']
                            changed = True
                            i += 1
                            continue
                    else:
                        # Merge into longer neighbor
                        prev_len = merged[-1]['end'] - merged[-1]['start'] if merged else 0
                        next_len = regimes[i + 1]['end'] - regimes[i + 1]['start']

                        if prev_len >= next_len and merged:
                            merged[-1]['end'] = seg['end']
                        else:
                            regimes[i + 1]['start'] = seg['start']
                        changed = True
                        i += 1
                        continue

                merged.append(seg)
                i += 1

            regimes = merged

            # Consolidate adjacent segments of same type
            consolidated = []
            for seg in regimes:
                if consolidated and consolidated[-1]['type'] == seg['type']:
                    consolidated[-1]['end'] = seg['end']
                else:
                    consolidated.append(seg)
            regimes = consolidated

        return regimes

    def classify_point_regimes(self, positions):
        """Return per-point regime labels (convenience method).

        Args:
            positions: (N, 2) array

        Returns:
            list of str, one per position: 'brownian', 'linear', or 'confined'
        """
        regimes = self.detect_regimes(positions)
        n = len(positions)
        labels = ['brownian'] * n
        for seg in regimes:
            for i in range(seg['start'], min(seg['end'], n)):
                labels[i] = seg['type']
        return labels


# ---------------------------------------------------------------------------
# MixedMotionPredictor  (plugin lines 955-1292)
# ---------------------------------------------------------------------------

class MixedMotionPredictor:
    """GPB1 mixed motion predictor with per-track filter banks.

    Maintains PARALLEL Kalman filters for three motion models (brownian,
    linear, confined) PER TRACK, and combines their predictions using
    model probabilities propagated through a 3x3 transition matrix.

    This is a faithful implementation of the plugin's GPB1 algorithm
    (Generalized Pseudo-Bayesian order 1).

    The predictor maintains:
      - _track_filters: Dict[int, Dict[str, KalmanFilter2D]]
            Per-track filter banks (one filter per motion model per track)
      - _track_model_probs: Dict[int, Dict[str, float]]
            Per-track model probabilities

    Args:
        dt: time step between frames
        process_noise_brownian: process noise for Brownian model
        process_noise_linear: process noise for linear model
        process_noise_confined: process noise for confined model
        measurement_noise: observation noise
        velocity_persistence: velocity persistence for linear model
        confinement_spring: spring constant for confined model
        transition_prob: probability of switching between any two models
    """

    # The three motion model names
    MODEL_NAMES = ['brownian', 'linear', 'confined']

    def __init__(self, dt=1.0, process_noise_brownian=1.0,
                 process_noise_linear=0.5, process_noise_confined=1.0,
                 measurement_noise=1.0, velocity_persistence=0.8,
                 confinement_spring=0.1, transition_prob=0.1):
        self.dt = dt
        self.process_noise_brownian = process_noise_brownian
        self.process_noise_linear = process_noise_linear
        self.process_noise_confined = process_noise_confined
        self.measurement_noise = measurement_noise
        self.velocity_persistence = velocity_persistence
        self.confinement_spring = confinement_spring

        # Per-track filter banks: track_id -> {model_name: KalmanFilter2D}
        self._track_filters = {}

        # Per-track model probabilities: track_id -> {model_name: float}
        self._track_model_probs = {}

        # 3x3 transition probability matrix
        # Rows = from model, Cols = to model
        n = len(self.MODEL_NAMES)
        p_stay = 1.0 - (n - 1) * transition_prob
        p_stay = max(p_stay, 0.01)  # safety clamp
        self._transition_matrix = np.full((n, n), transition_prob, dtype=np.float64)
        np.fill_diagonal(self._transition_matrix, p_stay)
        # Normalize rows
        self._transition_matrix /= self._transition_matrix.sum(axis=1, keepdims=True)

        # Legacy: global filters for backward compatibility
        self._global_filters = None
        self._global_model_probs = None

    def _make_filter(self, model_name, position=None, velocity=None):
        """Create a KalmanFilter2D for a specific motion model.

        Args:
            model_name: 'brownian', 'linear', or 'confined'
            position: optional initial position
            velocity: optional initial velocity

        Returns:
            KalmanFilter2D instance
        """
        if model_name == 'brownian':
            kf = KalmanFilter2D(
                dt=self.dt,
                process_noise=self.process_noise_brownian,
                measurement_noise=self.measurement_noise,
                motion_type='brownian')
        elif model_name == 'linear':
            kf = KalmanFilter2D(
                dt=self.dt,
                process_noise=self.process_noise_linear,
                measurement_noise=self.measurement_noise,
                motion_type='linear',
                velocity_persistence=self.velocity_persistence)
        elif model_name == 'confined':
            kf = KalmanFilter2D(
                dt=self.dt,
                process_noise=self.process_noise_confined,
                measurement_noise=self.measurement_noise,
                motion_type='confined',
                confinement_spring=self.confinement_spring,
                confinement_center=position)
        else:
            raise ValueError(f"Unknown model: {model_name!r}")

        if position is not None:
            kf.initialize(position, velocity)

        return kf

    # ------------------------------------------------------------------
    # Per-track filter bank management
    # ------------------------------------------------------------------

    def init_track_filters(self, track_id, position, velocity=None):
        """Initialize three parallel Kalman filters for a new track.

        Creates one KalmanFilter2D per motion model for the given track.

        Args:
            track_id: integer track identifier
            position: (2,) array initial position
            velocity: optional (2,) initial velocity
        """
        position = np.asarray(position, dtype=np.float64)
        filters = {}
        for name in self.MODEL_NAMES:
            filters[name] = self._make_filter(name, position, velocity)
        self._track_filters[track_id] = filters

        # Uniform initial model probabilities
        n = len(self.MODEL_NAMES)
        self._track_model_probs[track_id] = {
            name: 1.0 / n for name in self.MODEL_NAMES
        }

    def predict_gpb1(self, track_id, confinement_center=None):
        """GPB1 prediction step for a specific track.

        Runs predict() on each model's filter and returns per-model
        predicted states, covariances, and model probabilities.

        Args:
            track_id: integer track identifier
            confinement_center: optional (x, y) for confined model

        Returns:
            dict with:
                'states': {model_name: state_vector}
                'covariances': {model_name: covariance_matrix}
                'probabilities': {model_name: float}
                'position': weighted mean position (2,)
                'search_radius': max search radius across models
        """
        if track_id not in self._track_filters:
            raise KeyError(f"Track {track_id} not initialized. "
                           f"Call init_track_filters() first.")

        filters = self._track_filters[track_id]
        probs = self._track_model_probs[track_id]

        states = {}
        covariances = {}
        search_radii = []

        for name in self.MODEL_NAMES:
            kf = filters[name]
            if name == 'confined' and confinement_center is not None:
                x_pred, P_pred = kf.predict(confinement_center=confinement_center)
            else:
                x_pred, P_pred = kf.predict()
            states[name] = x_pred
            covariances[name] = P_pred
            search_radii.append(kf.get_search_radius())

        # Weighted mean position
        weighted_pos = np.zeros(2, dtype=np.float64)
        for name in self.MODEL_NAMES:
            weighted_pos += probs[name] * states[name][:2]

        return {
            'states': states,
            'covariances': covariances,
            'probabilities': dict(probs),
            'position': weighted_pos,
            'search_radius': float(max(search_radii))
        }

    def update_gpb1(self, track_id, measurement):
        """GPB1 update step for a specific track.

        Updates each model's filter with the measurement, computes
        innovation likelihoods, propagates through the transition matrix,
        and normalizes model probabilities.

        On regime change (dominant model switches), re-initializes the
        newly dominant filter to avoid large transients.

        Args:
            track_id: integer track identifier
            measurement: (2,) array observed [x, y]

        Returns:
            dict with:
                'position': weighted mean position (2,)
                'probabilities': {model_name: float}
                'dominant_model': str name of most probable model
                'likelihoods': {model_name: float}
        """
        if track_id not in self._track_filters:
            raise KeyError(f"Track {track_id} not initialized.")

        measurement = np.asarray(measurement, dtype=np.float64)
        filters = self._track_filters[track_id]
        probs = self._track_model_probs[track_id]

        # Previous dominant model
        prev_dominant = max(probs, key=probs.get)

        # Compute innovation likelihoods BEFORE update
        likelihoods = {}
        for name in self.MODEL_NAMES:
            likelihoods[name] = filters[name].innovation_likelihood(measurement)

        # Update each filter
        for name in self.MODEL_NAMES:
            filters[name].update(measurement)

        # GPB1 probability update through transition matrix
        prob_vec = np.array([probs[name] for name in self.MODEL_NAMES],
                            dtype=np.float64)
        like_vec = np.array([likelihoods[name] for name in self.MODEL_NAMES],
                            dtype=np.float64)

        # Predicted probabilities: c_j = sum_i (T_ij * mu_i)
        predicted_probs = self._transition_matrix.T @ prob_vec

        # Updated probabilities: mu_j = c_j * L_j / normalization
        updated = predicted_probs * like_vec
        total = np.sum(updated)
        if total > 1e-300:
            updated = updated / total
        else:
            # Fallback to uniform
            updated = np.ones(len(self.MODEL_NAMES)) / len(self.MODEL_NAMES)

        # Store back
        for i, name in enumerate(self.MODEL_NAMES):
            probs[name] = float(updated[i])

        # Detect regime change and re-initialize
        new_dominant = max(probs, key=probs.get)
        if new_dominant != prev_dominant:
            # Re-initialize the newly dominant filter at current measurement
            # to avoid large prediction errors from the stale state
            old_vel = filters[new_dominant].velocity
            filters[new_dominant].initialize(measurement, old_vel)
            logger.debug("Track %d regime change: %s -> %s",
                         track_id, prev_dominant, new_dominant)

        # Weighted mean position
        weighted_pos = np.zeros(2, dtype=np.float64)
        for name in self.MODEL_NAMES:
            weighted_pos += probs[name] * filters[name].position

        return {
            'position': weighted_pos,
            'probabilities': dict(probs),
            'dominant_model': new_dominant,
            'likelihoods': likelihoods
        }

    def cleanup_track(self, track_id):
        """Remove per-track filter banks to free memory.

        Args:
            track_id: integer track identifier
        """
        self._track_filters.pop(track_id, None)
        self._track_model_probs.pop(track_id, None)

    def get_track_dominant_model(self, track_id):
        """Return the dominant motion model for a track.

        Args:
            track_id: integer track identifier

        Returns:
            str: 'brownian', 'linear', or 'confined'
        """
        if track_id not in self._track_model_probs:
            return 'brownian'
        probs = self._track_model_probs[track_id]
        return max(probs, key=probs.get)

    def get_track_model_probs(self, track_id):
        """Return model probabilities for a track.

        Args:
            track_id: integer track identifier

        Returns:
            dict: {model_name: probability}
        """
        if track_id not in self._track_model_probs:
            n = len(self.MODEL_NAMES)
            return {name: 1.0 / n for name in self.MODEL_NAMES}
        return dict(self._track_model_probs[track_id])

    # ------------------------------------------------------------------
    # Legacy interface (global filters, no per-track management)
    # ------------------------------------------------------------------

    def initialize(self, position, velocity=None):
        """Initialize global filters (legacy interface).

        For backward compatibility. New code should use
        init_track_filters() instead.

        Args:
            position: (2,) array initial position
            velocity: optional (2,) initial velocity
        """
        position = np.asarray(position, dtype=np.float64)
        self._global_filters = {}
        for name in self.MODEL_NAMES:
            self._global_filters[name] = self._make_filter(name, position, velocity)

        n = len(self.MODEL_NAMES)
        self._global_model_probs = np.ones(n, dtype=np.float64) / n

    def predict(self):
        """Predict using global filters (legacy interface).

        Returns:
            dict with 'position', 'search_radius', 'model_probabilities',
                       'per_model_predictions'
        """
        if self._global_filters is None:
            raise RuntimeError("Global filters not initialized. "
                               "Call initialize() first.")

        predictions = {}
        for name in self.MODEL_NAMES:
            if name not in self._global_filters:
                continue
            kf = self._global_filters[name]
            x_pred, P_pred = kf.predict()
            predictions[name] = {'state': x_pred, 'covariance': P_pred}

        # Weighted position
        weighted_pos = np.zeros(2, dtype=np.float64)
        active_names = [n for n in self.MODEL_NAMES if n in self._global_filters]
        for i, name in enumerate(active_names):
            if i < len(self._global_model_probs):
                weighted_pos += self._global_model_probs[i] * predictions[name]['state'][:2]

        # Search radius (max across models)
        search_radius = max(
            kf.get_search_radius()
            for kf in self._global_filters.values()
        )

        prob_dict = {}
        for i, name in enumerate(active_names):
            if i < len(self._global_model_probs):
                prob_dict[name] = float(self._global_model_probs[i])

        return {
            'position': weighted_pos,
            'search_radius': search_radius,
            'model_probabilities': prob_dict,
            'per_model_predictions': predictions
        }

    def update(self, measurement):
        """Update global filters (legacy interface).

        Returns:
            weighted position (2,) array
        """
        if self._global_filters is None:
            raise RuntimeError("Global filters not initialized.")

        measurement = np.asarray(measurement, dtype=np.float64)
        active_names = [n for n in self.MODEL_NAMES if n in self._global_filters]

        likelihoods = np.zeros(len(active_names), dtype=np.float64)
        for i, name in enumerate(active_names):
            kf = self._global_filters[name]
            likelihoods[i] = kf.innovation_likelihood(measurement)
            kf.update(measurement)

        # GPB1 probability update
        n = len(active_names)
        if n > 0 and len(self._global_model_probs) >= n:
            prob_vec = self._global_model_probs[:n]
            T = self._transition_matrix[:n, :n]
            predicted_probs = T.T @ prob_vec
            updated = predicted_probs * likelihoods
            total = np.sum(updated)
            if total > 1e-300:
                self._global_model_probs[:n] = updated / total

        # Weighted state
        weighted_pos = np.zeros(2, dtype=np.float64)
        for i, name in enumerate(active_names):
            if i < len(self._global_model_probs):
                weighted_pos += self._global_model_probs[i] * self._global_filters[name].position

        return weighted_pos

    def predict_multiple_models(self):
        """Legacy fallback: predict from all global models.

        Returns list of (model_name, predicted_position, search_radius).
        """
        if self._global_filters is None:
            return []

        results = []
        for name in self.MODEL_NAMES:
            if name not in self._global_filters:
                continue
            kf = self._global_filters[name]
            x_pred, _ = kf.predict()
            r = kf.get_search_radius()
            results.append((name, x_pred[:2].copy(), r))
        return results

    def _update_model_weights(self, likelihoods_dict):
        """Legacy fallback: update global model weights from likelihood dict.

        Args:
            likelihoods_dict: {model_name: likelihood_value}
        """
        if self._global_filters is None:
            return

        active_names = [n for n in self.MODEL_NAMES if n in self._global_filters]
        n = len(active_names)
        if n == 0:
            return

        like_vec = np.array([likelihoods_dict.get(name, 1e-300)
                             for name in active_names], dtype=np.float64)
        prob_vec = self._global_model_probs[:n]
        T = self._transition_matrix[:n, :n]

        predicted = T.T @ prob_vec
        updated = predicted * like_vec
        total = np.sum(updated)
        if total > 1e-300:
            self._global_model_probs[:n] = updated / total

    @property
    def dominant_model(self):
        """Name of the most probable global motion model (legacy)."""
        if self._global_model_probs is None:
            return 'brownian'
        active_names = [n for n in self.MODEL_NAMES if n in (self._global_filters or {})]
        if not active_names:
            return 'brownian'
        idx = np.argmax(self._global_model_probs[:len(active_names)])
        return active_names[idx]
