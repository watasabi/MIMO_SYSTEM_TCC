"""Time series preprocessing: imputation, outlier removal, EWT-based
detrending, adaptive filtering, differencing and normalization, with a
position-aware inverse transform for evaluating model predictions.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from ewtpy import EWT1D, EWT_Meyer_FilterBank
from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ScalerType = Literal["standard", "minmax", "robust"]
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}
OutlierMethod = Literal[
    "iqr", "zscore", "mad", "quantile", "isolation_forest", "lof", "ecod"
]
ImputeMethod = Literal[
    "auto", "mean", "interpolate", "ffill", "bfill", "knn", "iterative"
]


def _ewt_with_boundaries(
    signal: npt.NDArray, boundaries: npt.NDArray
) -> npt.NDArray:
    """EWT of `signal` using fixed `boundaries` (as returned by `EWT1D`).

    `EWT1D` only detects boundaries on the signal it receives, so this
    reproduces its mirroring + Meyer filter-bank step with boundaries
    fitted elsewhere (e.g. on the training set).
    """
    half = int(np.ceil(signal.size / 2))
    mirrored = np.append(np.flip(signal[0 : half - 1]), signal)
    mirrored = np.append(mirrored, np.flip(signal[-half - 1 : -1]))
    spectrum = np.fft.fft(mirrored)
    bank = EWT_Meyer_FilterBank(boundaries, spectrum.size)
    ewt = np.zeros(bank.shape)
    for k in range(bank.shape[1]):
        ewt[:, k] = np.real(np.fft.ifft(np.conjugate(bank[:, k]) * spectrum))
    return ewt[half - 1 : -half, :]


class TimeSeriesPreprocessor:
    """Preprocessing pipeline for multivariate time series data.

    Applies, in order: log transform, imputation, outlier removal,
    EWT-based decomposition/detrending, adaptive (Savgol) filtering,
    differencing and normalization, then builds sliding-window
    sequences. `inverse_transform` undoes the position-dependent steps
    (differencing, detrend) per prediction window, so it can be used to
    evaluate model outputs against the original scale.
    """

    def __init__(
        self,
        normalize: bool = True,
        differencing: bool = False,
        detrend: bool = True,
        apply_ewt: bool = True,
        window_size: int = 24,
        horizon: int = 10,
        remove_outliers: bool = True,
        outlier_threshold: float = 0.05,
        outlier_method: OutlierMethod = "iqr",
        impute_method: ImputeMethod = "auto",
        ewt_bands: int = 5,
        trend_imf_idx: int = 0,
        log_transform: bool = False,
        filter_window: int = 5,
        filter_polyorder: int = 2,
        apply_filter: bool = True,
        scaler: ScalerType = "standard",
        step_size: int = 1,
        plot: bool = True,
    ) -> None:
        self.normalize = normalize
        self.scaler_type = scaler
        self.step_size = step_size
        self.plot = plot
        self.differencing = differencing
        self.detrend = detrend
        self.apply_ewt = apply_ewt
        self.window_size = window_size
        self.horizon = horizon
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.impute_method = impute_method
        self.ewt_bands = ewt_bands
        self.trend_imf_idx = trend_imf_idx
        self.log_transform = log_transform
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.apply_filter = apply_filter
        self.remove_outliers = remove_outliers

        # Fitted parameters
        self.scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self.log_offset: npt.NDArray | None = None
        self.diff_values: npt.NDArray | None = None
        self.pre_diff_values: npt.NDArray | None = None
        self.trend_component: npt.NDArray | None = None
        self.ewt_components: list[npt.NDArray] | None = None
        self.ewt_boundaries: list[npt.NDArray] | None = None

    def fit_transform(
        self,
        data: npt.NDArray,
        time_stamps: Sequence | None = None,
        feats: Sequence[int] | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Preprocess input data and return (X, y, full_processed_data).

        Args:
            data: Array of shape (n_timesteps, n_features).
            time_stamps: Optional timestamps, used for plots and to
                generate time features appended after normalization.
            feats: Column indices of `data` to keep as prediction
                targets in `y`. Defaults to all columns.

        Returns:
            Tuple `(X, y, processed)`: sliding windows, their targets,
            and the fully processed series.
        """
        processed = data.copy()

        if self.log_transform:
            min_val = processed.min(axis=0)
            self.log_offset = np.where(
                min_val <= 0, np.abs(min_val) + 1.0, 0.0
            )
            processed = np.log(processed + self.log_offset)

        # force a writable copy: pandas >= 3.0 Copy-on-Write makes
        # DataFrame.values read-only, which breaks the in-place
        # assignments below (differencing, detrend, ...)
        processed = np.array(self._impute_missing(processed), copy=True)
        self._plot_comparison(data, processed, "After Imputation", time_stamps)

        if self.remove_outliers:
            cleaned = np.stack(
                [
                    self._remove_outliers(processed[:, i]).ravel()
                    for i in range(processed.shape[1])
                ],
                axis=1,
            )
            self._plot_comparison(
                processed, cleaned, "After Outlier Removal", time_stamps
            )
            processed = cleaned

        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed, time_stamps)

        if self.apply_filter:
            filtered = self.adaptive_filter(processed)
            self._plot_comparison(
                processed, filtered, "After Adaptive Filtering", time_stamps
            )
            processed = filtered

        # kept for inverse_transform: values right before differencing,
        # in the same time index as self.trend_component
        self.pre_diff_values = processed.copy()

        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0

        if self.normalize:
            self.scaler = SCALERS[self.scaler_type]()
            processed = self.scaler.fit_transform(processed)

        if time_stamps is not None:
            time_feats = self.generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_feats), axis=1)

        return self._create_sequences(processed, feats) + (processed,)

    def transform(self, data: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Apply the same transformation as in fit_transform.

        Uses the parameters fitted on the training data (log offset, EWT
        boundaries, scaler); step order matches `fit_transform`.
        """
        return self._create_sequences(self.transform_series(data))

    def transform_series(self, data: npt.NDArray) -> npt.NDArray:
        """Transformed series (no windowing), same steps as `transform`.

        Useful for rolling/chained forecasts that slice their own
        windows; invert the outputs with `inverse_transform(...,
        reference=data)`.
        """
        if self.normalize and self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        pre_diff, _ = self._pre_diff_state(data)
        processed = pre_diff.copy()

        if self.differencing:
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0

        if self.normalize:
            processed = self.scaler.transform(processed)

        return processed

    def _pre_diff_state(
        self, data: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Run log/EWT-detrend/filter with fitted parameters.

        Returns:
            `(pre_diff_values, trend)` for `data`, in the same time index
            as `data`; `trend` is None when not detrending.
        """
        processed = np.array(data, dtype=float, copy=True)

        if self.log_transform:
            processed = np.log(processed + self.log_offset)

        trend = None
        if self.apply_ewt:
            if self.ewt_boundaries is None:
                raise ValueError("EWT not fitted. Call fit_transform first.")
            if self.detrend:
                trend = np.zeros_like(processed)
            for i in range(processed.shape[1]):
                ewt = _ewt_with_boundaries(
                    processed[:, i], self.ewt_boundaries[i]
                )
                if self.detrend:
                    trend[:, i] = ewt[:, self.trend_imf_idx]
                    processed[:, i] -= trend[:, i]

        if self.apply_filter:
            processed = self.adaptive_filter(processed)

        return processed, trend

    def anchor_indices(self, n_windows: int) -> npt.NDArray:
        """Row of the first predicted step of each window.

        This is the `anchor_indices` argument of `inverse_transform`
        for the windows returned by `fit_transform`/`transform`.
        """
        return np.arange(n_windows) * self.step_size + self.window_size

    def inverse_transform(  # noqa: PLR0914
        self,
        predictions: npt.NDArray,
        anchor_indices: npt.NDArray,
        feats: Sequence[int] | None = None,
        reference: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Invert a batch of predicted windows back to the original scale.

        Undoes, in reverse pipeline order, only the steps whose effect
        depends on absolute time position (differencing, detrend) using
        each window's own anchor, plus the elementwise steps (normalize,
        log). Outlier removal, EWT band splitting and the Savgol filter
        are lossy and are not inverted.

        Args:
            predictions: Array of shape (n_samples, horizon, n_cols)
                with model outputs in the space returned by
                `fit_transform`/`transform` (i.e. before the time
                features that may have been concatenated).
            anchor_indices: 1D array of length n_samples. For sample k,
                `anchor_indices[k]` is the row of the first predicted
                step of that window, in the time index of `reference`
                (or of the data given to `fit_transform` when
                `reference` is None). See `anchor_indices()`.
            feats: Column indices (of the original data) that the
                `n_cols` prediction columns correspond to, e.g. the
                target columns. Defaults to all columns.
            reference: Raw segment the windows were built from (the
                array given to `transform`). When set, differencing and
                detrend are undone with this segment's own values and
                EWT trend, so the round trip is exact. When None, the
                fitted (training) state is used and the trend is
                linearly extrapolated past its end.

        Returns:
            Array of shape (n_samples, horizon, n_cols) in the original
            data scale.
        """
        predictions = np.asarray(predictions, dtype=float).copy()
        anchor_indices = np.asarray(anchor_indices)
        n_samples, horizon, n_cols = predictions.shape
        n_features = len(self.scaler.scale_) if self.scaler else n_cols
        feats = list(range(n_cols)) if feats is None else list(feats)

        if reference is not None:
            pre_diff, trend = self._pre_diff_state(reference)
        else:
            pre_diff, trend = self.pre_diff_values, self.trend_component

        if self.normalize:
            # Scalers are per column: pad to full width, invert, select.
            padded = np.zeros((n_samples * horizon, n_features))
            padded[:, feats] = predictions.reshape(-1, n_cols)
            predictions = (
                self.scaler.inverse_transform(padded)[:, feats]
            ).reshape(n_samples, horizon, n_cols)

        if self.differencing:
            if pre_diff is None:
                raise ValueError(
                    "pre_diff_values not fitted. Call fit_transform first."
                )
            base = pre_diff[anchor_indices - 1][:, feats]
            predictions = base[:, None, :] + np.cumsum(predictions, axis=1)

        if self.detrend and trend is not None:
            trend = trend[:, feats]
            trend_len = len(trend)
            trend_to_add = np.zeros_like(predictions)
            for k, anchor in enumerate(anchor_indices):
                end = anchor + horizon
                if end <= trend_len:
                    trend_to_add[k] = trend[anchor:end]
                else:
                    in_range = max(trend_len - anchor, 0)
                    if in_range > 0:
                        trend_to_add[k, :in_range] = trend[
                            anchor : anchor + in_range
                        ]
                    tail = min(10, trend_len)
                    slope = (trend[-1] - trend[-tail]) / tail
                    for j in range(in_range, horizon):
                        steps_beyond = j - in_range + 1
                        trend_to_add[k, j] = trend[-1] + slope * steps_beyond
            predictions += trend_to_add

        if self.log_transform:
            predictions = np.exp(predictions) - self.log_offset[feats]

        return predictions

    def _remove_outliers(self, data_col: npt.NDArray) -> npt.NDArray:
        method = self.outlier_method
        threshold = self.outlier_threshold
        x = data_col.reshape(-1, 1)

        if method == "iqr":
            Q1, Q3 = np.percentile(x, [25, 75], axis=0)
            IQR = Q3 - Q1
            return np.clip(x, Q1 - threshold * IQR, Q3 + threshold * IQR)

        elif method == "zscore":
            return np.clip(
                x,
                x.mean() - threshold * x.std(),
                x.mean() + threshold * x.std(),
            )

        elif method == "mad":
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            return np.where(
                np.abs((x - med) / (mad + 1e-6)) <= threshold * 1.4826, x, med
            )

        elif method == "quantile":
            low, high = np.percentile(
                x, [threshold * 100, 100 - threshold * 100]
            )
            return np.clip(x, low, high)

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            pred = IsolationForest(contamination=threshold).fit_predict(x)
            return np.where(pred == 1, x, np.nan)

        elif method == "lof":
            pred = LocalOutlierFactor(
                n_neighbors=20, contamination=threshold
            ).fit_predict(x)
            return np.where(pred == 1, x, np.nan)

        elif method == "ecod":
            from pyod.models.ecod import ECOD

            pred = ECOD().fit(x).predict(x)
            return np.where(pred == 0, x, np.nan)

        raise ValueError(f"Unsupported outlier method: {method}")

    def _impute_missing(self, data: npt.NDArray) -> npt.NDArray:
        df = pd.DataFrame(data)
        method = self.impute_method

        if method == "auto":
            filled = df.copy()
            for col in df.columns:
                if df[col].isna().mean() < 0.05:
                    filled[col] = df[col].interpolate().ffill().bfill()
                elif df[col].isna().mean() < 0.2:
                    filled[col] = (
                        KNNImputer(n_neighbors=3)
                        .fit_transform(df[[col]])
                        .ravel()
                    )
                else:
                    try:
                        from fancyimpute import IterativeImputer

                        filled[col] = (
                            IterativeImputer().fit_transform(df[[col]]).ravel()
                        )
                    except ImportError:
                        filled[col] = df[col].fillna(df[col].mean())
            return filled.values

        if method == "mean":
            return df.fillna(df.mean()).values
        elif method == "interpolate":
            return df.interpolate().ffill().bfill().values
        elif method == "ffill":
            return df.ffill().bfill().values
        elif method == "bfill":
            return df.bfill().ffill().values
        elif method == "knn":
            return KNNImputer(n_neighbors=5).fit_transform(df)
        elif method == "iterative":
            try:
                from fancyimpute import IterativeImputer

                return IterativeImputer().fit_transform(df)
            except ImportError:
                warnings.warn("fancyimpute not installed, using mean fallback")
                return df.fillna(df.mean()).values

        raise ValueError(f"Unsupported imputation method: {method}")

    def _apply_ewt_and_detrend(
        self, data: npt.NDArray, time_stamps: Sequence | None
    ) -> npt.NDArray:
        self.ewt_components = []
        self.ewt_boundaries = []
        if self.detrend:
            self.trend_component = np.zeros_like(data)

        for i in range(data.shape[1]):
            signal = data[:, i]
            ewt, _, bounds = EWT1D(signal, N=self.ewt_bands)
            self.ewt_components.append(ewt)
            self.ewt_boundaries.append(bounds)

            if self.detrend:
                trend = ewt[:, self.trend_imf_idx]
                self.trend_component[:, i] = trend
                data[:, i] = signal - trend
        return data

    def adaptive_filter(self, data: npt.NDArray) -> npt.NDArray:
        """Smooth `data` along the time axis with a Savgol filter."""
        return savgol_filter(
            data, self.filter_window, self.filter_polyorder, axis=0
        )

    def generate_time_features(
        self, timestamps: Sequence, freq: str = "h"
    ) -> npt.NDArray:
        """Build normalized calendar features (month/day/weekday/hour)."""
        df = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
        df["month"] = df.ts.dt.month / 12.0
        df["day"] = df.ts.dt.day / 31.0
        df["weekday"] = df.ts.dt.weekday / 6.0
        df["hour"] = df.ts.dt.hour / 23.0 if freq.lower() == "h" else 0.0
        return df[["month", "day", "weekday", "hour"]].values.astype(
            np.float32
        )

    def _create_sequences(
        self, data: npt.NDArray, feats: Sequence[int] | None = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        feats = list(range(data.shape[1])) if feats is None else feats
        X, y = [], []
        last = len(data) - self.window_size - self.horizon + 1
        for i in range(0, last, self.step_size):
            X.append(data[i : i + self.window_size])
            y.append(
                data[
                    i + self.window_size : i + self.window_size + self.horizon
                ][:, feats]
            )
        return np.array(X), np.array(y)

    def _plot_comparison(
        self,
        original: npt.NDArray,
        cleaned: npt.NDArray,
        title: str = "Preprocessing Comparison",
        time_stamps: Sequence | None = None,
    ) -> None:
        if not self.plot:
            return
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)
        if original.shape[0] == 1:
            original = original.T
        if cleaned.shape[0] == 1:
            cleaned = cleaned.T

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        x = (
            time_stamps
            if time_stamps is not None
            else np.arange(original.shape[0])
        )

        for i in range(original.shape[1]):
            axs[0].plot(x, original[:, i], label=f"Feature {i}")
            axs[1].plot(x, cleaned[:, i], label=f"Feature {i}")
        axs[0].set_title("Original")
        axs[1].set_title("Cleaned")
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_ewt_components(self) -> list[npt.NDArray] | None:
        """Return the per-feature EWT band decomposition, if computed."""
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self) -> npt.NDArray | None:
        """Return the detrended trend component, if computed."""
        return self.trend_component if self.detrend else None
