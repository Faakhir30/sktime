# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TabICL tabular foundation model forecaster."""

__author__ = ["fkiraly"]
__all__ = ["TabICLForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class TabICLForecaster(BaseForecaster):
    r"""TabICL tabular foundation model forecaster for zero-shot TS forecasting.

    Direct interface to ``TabICLForecaster`` from the ``tabicl`` package [1]_.

    TabICL [2]_ is a tabular in-context learning foundation model. Its forecaster
    uses the regression backbone (``TabICLRegressor``) with a sliding-window
    tabular encoding of the time series: each prediction target is predicted from
    a window of lagged values (and optionally temporal features), formulated as a
    tabular regression problem. The result is a powerful zero-shot forecaster that
    requires no fine-tuning and operates without fitting model weights to new data.

    Key properties:
    - Zero-shot: no training required, applies ICL at inference time.
    - State-of-the-art accuracy on TabArena and TALENT benchmarks.
    - Supports univariate time series forecasting.
    - Uses the same pre-trained tabular regression checkpoint as ``TabICLRegressor``.

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of historical timesteps to use as model context.
        Longer context may improve accuracy but increases memory and runtime.
    temporal_features : str, list of str, or None, default=None
        Timestep index, calendar patterns, and seasonality features to include.
        If None, no temporal features are added. If ``"auto"``, TabICL selects
        features automatically based on the detected frequency.
    point_estimate : str, default="mean"
        Method for point prediction. One of ``"mean"`` or ``"median"``.
    n_estimators : int, default=8
        Number of ensemble members. More members improve accuracy but increase
        runtime linearly.
    norm_methods : list of str or None, default=None
        Normalization methods to try. If None, defaults to TabICL built-in choices.
    feat_shuffle_method : str, default="latin"
        Feature permutation strategy for ensemble diversity.
    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping.
    batch_size : int, default=8
        Number of ensemble members processed together. Lower to save GPU memory.
    kv_cache : bool, default=False
        If True, cache key-value projections of training data for faster repeated
        inference. Requires more memory.
    model_path : str or None, default=None
        Path to a local checkpoint file. If None, downloads from Hugging Face.
    allow_auto_download : bool, default=True
        If True, automatically download the checkpoint if not found locally.
    checkpoint_version : str or None, default=None
        Pretrained checkpoint version string. If None, TabICL uses its default
        (currently ``"tabicl-regressor-v2-20260212.ckpt"``).
    device : str or None, default=None
        Inference device. If None, auto-selects CUDA or CPU.
        Use ``"mps"`` for Apple Silicon.
    use_amp : str or bool, default="auto"
        Automatic mixed precision setting for faster inference.
    use_fa3 : str or bool, default="auto"
        Flash Attention 3 setting for Hopper GPUs (e.g., H100).
    offload_mode : str, default="auto"
        Decides when to use CPU/disk offloading.
    disk_offload_dir : str or None, default=None
        Directory for disk offloading. If None, uses a temp directory.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int or None, default=None
        Number of PyTorch threads for CPU inference. If None, uses PyTorch default.
    verbose : bool, default=False
        If True, print detailed information during inference.
    inference_config : dict or None, default=None
        Fine-grained inference control dict for advanced users.
        Passed to the underlying ``TabICLRegressor``.

    Attributes
    ----------
    tabicl_forecaster_ : tabicl.TabICLForecaster
        The underlying TabICL forecaster instance (instantiated at predict time).

    See Also
    --------
    sktime.classification.foundation_models.tabicl.TabICLClassifier :
        TabICL adapter for time series classification.
    sktime.regression.foundation_models.tabicl.TabICLRegressor :
        TabICL adapter for time series regression.

    References
    ----------
    .. [1] https://github.com/soda-inria/tabicl
    .. [2] @article{qu2026tabiclv2,
            title={{TabICLv2}: {A} better, faster, scalable, and
            open tabular foundation model},
            author={Qu, Jingang and Holzm{\"u}ller, David and
            Varoquaux, Ga{\"e}l and Le Morvan, Marine},
            journal={arXiv preprint arXiv:2602.11139},
            year={2026}
            }

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabicl import TabICLForecaster
    >>> y = load_airline()
    >>> forecaster = TabICLForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    TabICLForecaster()  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Jingang", "dholzmueller", "marineLM", "Faakhir30"],
        "maintainers": ["Faakhir30"],
        "python_dependencies": ["tabicl[forecast]"],
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "None",
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:random_state": True,
        # testing flags
        # -------------
        "tests:vm": True,
    }

    def __init__(
        self,
        max_context_length=4096,
        temporal_features=None,
        point_estimate="mean",
        n_estimators=8,
        norm_methods=None,
        feat_shuffle_method="latin",
        outlier_threshold=4.0,
        batch_size=8,
        kv_cache=False,
        model_path=None,
        allow_auto_download=True,
        checkpoint_version=None,
        device=None,
        use_amp="auto",
        use_fa3="auto",
        offload_mode="auto",
        disk_offload_dir=None,
        random_state=42,
        n_jobs=None,
        verbose=False,
        inference_config=None,
    ):
        self.max_context_length = max_context_length
        self.temporal_features = temporal_features
        self.point_estimate = point_estimate
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.inference_config = inference_config

        super().__init__()

    def _get_tabicl_config(self):
        """Build the tabicl_config dict for the underlying TabICLRegressor.

        Returns
        -------
        tabicl_config : dict
            Keyword arguments for ``tabicl.TabICLRegressor``.
        """
        config = {
            "n_estimators": self.n_estimators,
            "feat_shuffle_method": self.feat_shuffle_method,
            "outlier_threshold": self.outlier_threshold,
            "batch_size": self.batch_size,
            "kv_cache": self.kv_cache,
            "allow_auto_download": self.allow_auto_download,
            "use_amp": self.use_amp,
            "use_fa3": self.use_fa3,
            "offload_mode": self.offload_mode,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
        # only pass optional args if set, to respect TabICL's own defaults
        if self.norm_methods is not None:
            config["norm_methods"] = self.norm_methods
        if self.model_path is not None:
            config["model_path"] = self.model_path
        if self.checkpoint_version is not None:
            config["checkpoint_version"] = self.checkpoint_version
        if self.device is not None:
            config["device"] = self.device
        if self.disk_offload_dir is not None:
            config["disk_offload_dir"] = self.disk_offload_dir
        if self.n_jobs is not None:
            config["n_jobs"] = self.n_jobs
        if self.inference_config is not None:
            config["inference_config"] = self.inference_config
        return config

    def _get_tabicl_forecaster_kwargs(self):
        """Build keyword arguments for TabICLForecaster instantiation.

        Returns
        -------
        kwargs : dict
            Keyword arguments for ``tabicl.TabICLForecaster``.
        """
        return {
            "max_context_length": self.max_context_length,
            "temporal_features": self.temporal_features,
            "point_estimate": self.point_estimate,
            "tabicl_config": self._get_tabicl_config(),
        }

    @staticmethod
    def _y_to_context_df(y, item_id="series_0"):
        """Convert a sktime pd.Series y to tabicl context DataFrame format.

        Parameters
        ----------
        y : pd.Series
            Time series with a datetime-like or Period index.
        item_id : str, default="series_0"
            Item identifier to use in the context DataFrame.

        Returns
        -------
        context_df : pd.DataFrame
            DataFrame with columns ``item_id``, ``timestamp``, ``target``.
        """
        index = y.index
        # convert PeriodIndex to Timestamps (end of period)
        if hasattr(index, "to_timestamp"):
            timestamps = index.to_timestamp()
        else:
            timestamps = index

        context_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": timestamps,
                "target": y.values,
            }
        )
        return context_df

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        For TabICL, fitting amounts to storing the training data; the actual
        in-context learning is performed during prediction.

        Parameters
        ----------
        y : pd.Series
            Target time series.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : reference to self.
        """
        # TabICL requires no model fitting - the base class stores y as self._y,
        # which will be used as the in-context training data at predict time.
        # The underlying TabICLForecaster is instantiated fresh at each predict call
        # (it is stateless: no weights are learned, ICL is done in the forward pass).
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.Series
            Point forecasts at ``fh``, indexed by the absolute forecast timestamps.
        """
        from tabicl import TabICLForecaster as _TabICLForecaster

        # Determine how many steps ahead to predict (take the furthest step)
        prediction_length = int(max(fh.to_relative(self.cutoff)))

        # Build context DataFrame from training data stored by the base class
        context_df = self._y_to_context_df(self._y, item_id="series_0")

        # Instantiate and call the TabICL forecaster
        tabicl_fc = _TabICLForecaster(**self._get_tabicl_forecaster_kwargs())
        pred_df = tabicl_fc.predict_df(context_df, prediction_length=prediction_length)

        # pred_df has columns: item_id, timestamp, mean (or median)
        # Filter to our single series and sort by timestamp for safety
        pred_df = pred_df[pred_df["item_id"] == "series_0"].sort_values("timestamp")
        pred_values = pred_df[self.point_estimate].values  # shape (prediction_length,)

        # Map to absolute fh timestamps using ForecastingHorizon
        fh_abs = fh.to_absolute(self.cutoff).to_pandas()

        # fh steps are 1-based relative indices; select from pred_values
        fh_relative = fh.to_relative(self.cutoff)
        # relative indices are 1-based; pred_values is 0-based
        fh_idx = np.array([int(r) - 1 for r in fh_relative])
        selected_values = pred_values[fh_idx]

        y_pred = pd.Series(selected_values, index=fh_abs, name=self._y.name)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {}
        params2 = {
            "n_estimators": 2,
            "batch_size": 2,
            "verbose": False,
        }

        return [params1, params2]
