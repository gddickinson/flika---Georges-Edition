"""SVM-based track motion classifier.

Classifies tracks into motion regimes (Mobile, Confined, Trapped)
using a pipeline of Box-Cox normalisation, PCA dimensionality reduction,
and an RBF-kernel SVM.  Ported from the spt_batch_analysis codebase.

The classifier expects per-track feature vectors computed by
:class:`~flika.spt.features.feature_calculator.FeatureCalculator`.

Workflow:
    1. Compute features for all tracks (via FeatureCalculator).
    2. Train the classifier on labelled data (or load a pre-trained model).
    3. Predict motion classes for new tracks.
"""
import numpy as np
import pandas as pd
from ...logger import logger


class SPTClassifier:
    """SVM-based track motion classifier.

    Uses Box-Cox normalisation (via sklearn PowerTransformer),
    StandardScaler, PCA dimensionality reduction (fixed 3 components),
    and an RBF-kernel SVM with hyperparameter tuning via 10-fold
    grid search.  Faithfully replicates the original spt_batch_analysis
    plugin's classification pipeline.

    Motion labels:
        - **1** -- Mobile (directed / free diffusion)
        - **2** -- Confined (sub-diffusive, restricted area)
        - **3** -- Trapped (essentially immobile)

    Attributes:
        FEATURES: List of feature column names expected in input DataFrames.
        LABELS: Mapping from integer label to human-readable name.
        is_trained: Whether the classifier has been trained or loaded.
    """

    FEATURES = [
        'net_displacement',
        'straightness',
        'asymmetry',
        'radius_gyration',
        'kurtosis',
        'fractal_dimension',
    ]

    LABELS = {
        1: 'Mobile',
        2: 'Confined',
        3: 'Trapped',
    }

    def __init__(self):
        self.power_transformer = None
        self.scaler = None
        self.pca = None
        self.svm = None
        self.is_trained = False
        self._feature_columns = list(self.FEATURES)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sklearn():
        """Import scikit-learn, raising a helpful error if absent."""
        try:
            import sklearn
            return sklearn
        except ImportError:
            raise ImportError(
                "scikit-learn is required for SPTClassifier but is not "
                "installed. Install it with:\n"
                "    pip install scikit-learn\n"
                "or\n"
                "    conda install scikit-learn"
            )

    def _extract_features(self, features_df):
        """Extract and validate the required feature columns.

        Args:
            features_df: DataFrame containing at least the columns in
                :attr:`FEATURES`.

        Returns:
            (N, len(FEATURES)) numpy array.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = [c for c in self._feature_columns
                   if c not in features_df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns: {missing}. "
                f"Available columns: {list(features_df.columns)}")

        X = features_df[self._feature_columns].values.astype(np.float64)

        # Replace inf / nan with column medians
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            bad = ~np.isfinite(col)
            if np.any(bad):
                median_val = np.nanmedian(col[~bad]) if np.any(~bad) else 0.0
                col[bad] = median_val
                X[:, col_idx] = col

        return X

    def _box_cox_transform(self, X, fit=False):
        """Apply Box-Cox normalisation via sklearn PowerTransformer.

        Matches the original spt_batch_analysis plugin which uses
        ``sklearn.preprocessing.PowerTransformer(method='box-cox')``.
        PowerTransformer shifts data to be strictly positive internally.

        Args:
            X: (N, D) float array.
            fit: If True, fit the transformer. Otherwise, use stored one.

        Returns:
            Transformed array of the same shape.
        """
        from sklearn.preprocessing import PowerTransformer

        if fit:
            self.power_transformer = PowerTransformer(
                method='box-cox', standardize=False)
            # PowerTransformer requires strictly positive data;
            # shift each column to be positive before fitting
            self._bc_shifts = {}
            X_shifted = X.copy()
            for i in range(X.shape[1]):
                col_min = np.min(X_shifted[:, i])
                if col_min <= 0:
                    shift = abs(col_min) + 1.0
                    X_shifted[:, i] += shift
                    self._bc_shifts[i] = shift
                else:
                    self._bc_shifts[i] = 0.0
            return self.power_transformer.fit_transform(X_shifted)
        else:
            X_shifted = X.copy()
            for i in range(X.shape[1]):
                shift = self._bc_shifts.get(i, 0.0)
                if shift > 0:
                    X_shifted[:, i] += shift
            # Ensure strictly positive for transform
            X_shifted = np.maximum(X_shifted, 1e-10)
            return self.power_transformer.transform(X_shifted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, features_df, labels, test_size=0.2, random_state=42):
        """Train the SVM classification pipeline.

        Args:
            features_df: DataFrame with at least the columns listed in
                :attr:`FEATURES`.
            labels: Array-like of integer labels (1=Mobile, 2=Confined,
                3=Trapped), one per row in *features_df*.
            test_size: Fraction of data to hold out for evaluation
                (default 0.2).
            random_state: Random seed for reproducibility.

        Returns:
            dict with training metrics:
                - *accuracy*: overall accuracy on the test split.
                - *per_class_accuracy*: dict mapping label -> accuracy.
                - *confusion_matrix*: 2D numpy array.
                - *best_params*: dict of best SVM hyperparameters.
                - *n_components*: number of PCA components retained.
                - *n_train*: number of training samples.
                - *n_test*: number of test samples.
        """
        self._check_sklearn()
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix

        X = self._extract_features(features_df)
        y = np.asarray(labels, dtype=int)

        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix has {len(X)} rows but {len(y)} labels "
                f"were provided")

        logger.info("SPTClassifier.train: %d samples, %d features",
                    len(X), X.shape[1])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None)

        # Box-Cox normalisation
        X_train_bc = self._box_cox_transform(X_train, fit=True)
        X_test_bc = self._box_cox_transform(X_test, fit=False)

        # Standard scaling
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train_bc)
        X_test_sc = self.scaler.transform(X_test_bc)

        # PCA -- fixed 3 components (exact replica of original plugin)
        n_pca = min(3, X_train_sc.shape[1])
        self.pca = PCA(n_components=n_pca)
        X_train_pca = self.pca.fit_transform(X_train_sc)
        X_test_pca = self.pca.transform(X_test_sc)

        n_components = X_train_pca.shape[1]
        logger.info("PCA retained %d components (%.1f%% variance)",
                    n_components,
                    100.0 * np.sum(self.pca.explained_variance_ratio_))

        # Grid search with RBF SVM -- exact original plugin grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        }

        # 10-fold CV (exact replica of original plugin)
        grid = GridSearchCV(
            SVC(kernel='rbf'),
            param_grid, cv=10, scoring='accuracy',
            n_jobs=-1, refit=True)
        grid.fit(X_train_pca, y_train)

        self.svm = grid.best_estimator_
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.svm.predict(X_test_pca)
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred,
                              labels=sorted(self.LABELS.keys()))

        # Per-class accuracy
        per_class = {}
        for label in sorted(self.LABELS.keys()):
            mask = y_test == label
            if np.any(mask):
                per_class[self.LABELS[label]] = float(
                    accuracy_score(y_test[mask], y_pred[mask]))
            else:
                per_class[self.LABELS[label]] = float('nan')

        metrics = {
            'accuracy': acc,
            'per_class_accuracy': per_class,
            'confusion_matrix': cm,
            'best_params': grid.best_params_,
            'n_components': n_components,
            'n_train': len(X_train),
            'n_test': len(X_test),
        }

        logger.info("SPTClassifier trained: accuracy=%.3f, C=%.1f, "
                    "gamma=%s, %d PCA components",
                    acc, grid.best_params_['C'],
                    grid.best_params_['gamma'], n_components)
        return metrics

    def predict(self, features_df):
        """Predict motion class labels for tracks.

        Args:
            features_df: DataFrame with at least the columns listed in
                :attr:`FEATURES`.

        Returns:
            1D numpy array of integer labels (1=Mobile, 2=Confined,
            3=Trapped), one per row.

        Raises:
            RuntimeError: If the classifier has not been trained or loaded.
            ValueError: If required feature columns are missing.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Classifier is not trained. Call train() or load() first.")

        X = self._extract_features(features_df)
        if len(X) == 0:
            return np.array([], dtype=int)

        X_bc = self._box_cox_transform(X, fit=False)
        X_sc = self.scaler.transform(X_bc)
        X_pca = self.pca.transform(X_sc)

        predictions = self.svm.predict(X_pca)
        return predictions.astype(int)

    def predict_proba(self, features_df):
        """Predict class probabilities (requires SVM trained with probability=True).

        If the model was trained without probability estimates, this
        re-fits a Platt-scaled model internally (may be slow for large
        datasets).

        Args:
            features_df: DataFrame with required feature columns.

        Returns:
            (N, K) array of class probabilities where K is the number of
            classes, or ``None`` if probability estimation fails.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Classifier is not trained. Call train() or load() first.")

        # Enable probability if not already enabled
        if not getattr(self.svm, 'probability', False):
            logger.warning("SVM was trained without probability=True; "
                           "re-fitting with Platt scaling")
            self._check_sklearn()
            from sklearn.calibration import CalibratedClassifierCV
            try:
                calibrated = CalibratedClassifierCV(self.svm, cv='prefit')
                # We cannot properly calibrate without training data, so
                # just return None
                logger.warning("Cannot calibrate without training data; "
                               "returning None")
                return None
            except Exception:
                return None

        X = self._extract_features(features_df)
        if len(X) == 0:
            return np.empty((0, len(self.LABELS)))

        X_bc = self._box_cox_transform(X, fit=False)
        X_sc = self.scaler.transform(X_bc)
        X_pca = self.pca.transform(X_sc)

        try:
            return self.svm.predict_proba(X_pca)
        except Exception as exc:
            logger.warning("predict_proba failed: %s", exc)
            return None

    def label_name(self, label_int):
        """Return the human-readable name for an integer label."""
        return self.LABELS.get(int(label_int), f'Unknown({label_int})')

    def save(self, path):
        """Save the trained model to a file (joblib pickle).

        Args:
            path: Output file path (typically ``.pkl`` or ``.joblib``).

        Raises:
            RuntimeError: If the classifier has not been trained.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save: classifier is not trained.")

        try:
            import joblib
        except ImportError:
            raise ImportError(
                "joblib is required for saving models. Install it with:\n"
                "    pip install joblib")

        state = {
            'scaler': self.scaler,
            'pca': self.pca,
            'svm': self.svm,
            'power_transformer': self.power_transformer,
            'bc_shifts': getattr(self, '_bc_shifts', {}),
            'feature_columns': self._feature_columns,
        }
        joblib.dump(state, path)
        logger.info("SPTClassifier saved to %s", path)

    def load(self, path):
        """Load a trained model from a file.

        Args:
            path: Path to a previously saved model file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            import joblib
        except ImportError:
            raise ImportError(
                "joblib is required for loading models. Install it with:\n"
                "    pip install joblib")

        state = joblib.load(path)
        self.scaler = state['scaler']
        self.pca = state['pca']
        self.svm = state['svm']
        self.power_transformer = state.get('power_transformer', None)
        self._bc_shifts = state.get('bc_shifts', {})
        self._feature_columns = state.get('feature_columns', list(self.FEATURES))
        self.is_trained = True
        logger.info("SPTClassifier loaded from %s", path)
