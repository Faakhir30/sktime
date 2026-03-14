"""TabICL tabular foundation model time series regressor."""

__author__ = ["fkiraly"]
__all__ = ["TabICLRegressor"]


from sktime.regression.base import BaseRegressor


class TabICLRegressor(BaseRegressor):
    r"""TabICL tabular foundation model time series regressor.

    Direct interface to ``TabICLRegressor`` from the ``tabicl`` package [1]_.

    TabICL [2]_ is a tabular in-context learning foundation model that performs
    regression via a single forward pass through a pre-trained transformer.
    It stores training data in-context and makes predictions at inference time.

    To use for time series regression, each time series instance is flattened to
    a 1D feature vector of length ``n_dims * n_timepoints``, which is then passed
    to the TabICL tabular model. For multivariate time series, dimensions are
    concatenated along the feature axis.

    Parameters
    ----------
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
        Regression was introduced in v2.0, so only v2 checkpoint is supported.
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
        Passed directly to the underlying ``TabICLRegressor``.

    Attributes
    ----------
    tabicl_regressor_ : tabicl.TabICLRegressor
        The fitted underlying TabICL regressor.

    References
    ----------
    .. [1] https://github.com/soda-inria/tabicl
    .. [2] @article{qu2026tabiclv2,
            title={{TabICLv2}: {A} better, faster, scalable, and
            open tabular foundation model},
            author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux,
            Ga{\"e}l and Le Morvan, Marine},
            journal={arXiv preprint arXiv:2602.11139},
            year={2026}
            }

    Examples
    --------
    >>> from sktime.regression.foundation_models.tabicl import TabICLRegressor
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    >>> X_test, _ = load_unit_test(split="test", return_type="numpy3d")
    >>> reg = TabICLRegressor()  # doctest: +SKIP
    >>> reg.fit(X_train, y_train.astype(float))  # doctest: +SKIP
    TabICLRegressor(...)  # doctest: +SKIP
    >>> y_pred = reg.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Jingang", "dholzmueller", "marineLM", "Faakhir30"],
        "maintainers": ["Faakhir30"],
        "python_dependencies": ["tabicl"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:unequal_length": False,
        "capability:multithreading": False,
        "capability:random_state": True,
        # testing flags
        # -------------
        "tests:vm": True,
    }

    def __init__(
        self,
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

    def _get_tabicl_kwargs(self):
        """Collect kwargs for TabICLRegressor instantiation.

        Returns
        -------
        kwargs : dict
            Keyword arguments for ``tabicl.TabICLRegressor``.
        """
        kwargs = {
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
            kwargs["norm_methods"] = self.norm_methods
        if self.model_path is not None:
            kwargs["model_path"] = self.model_path
        if self.checkpoint_version is not None:
            kwargs["checkpoint_version"] = self.checkpoint_version
        if self.device is not None:
            kwargs["device"] = self.device
        if self.disk_offload_dir is not None:
            kwargs["disk_offload_dir"] = self.disk_offload_dir
        if self.n_jobs is not None:
            kwargs["n_jobs"] = self.n_jobs
        if self.inference_config is not None:
            kwargs["inference_config"] = self.inference_config
        return kwargs

    @staticmethod
    def _panel_to_tabular(X):
        """Flatten 3D panel array to 2D tabular array.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)
            Time series panel data.

        Returns
        -------
        X_tab : np.ndarray of shape (n_instances, n_dims * n_timepoints)
            Tabular representation of the panel data.
        """
        n_instances, n_dims, n_timepoints = X.shape
        return X.reshape(n_instances, n_dims * n_timepoints)

    def _fit(self, X, y):
        """Fit the regressor.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)
            Training time series panel data.
        y : np.ndarray of shape (n_instances,)
            Training target values (continuous).

        Returns
        -------
        self : reference to self.
        """
        from tabicl import TabICLRegressor as _TabICLRegressor

        X_tab = self._panel_to_tabular(X)
        self.tabicl_regressor_ = _TabICLRegressor(**self._get_tabicl_kwargs())
        self.tabicl_regressor_.fit(X_tab, y)
        return self

    def _predict(self, X):
        """Predict regression values for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)
            Test time series panel data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_instances,)
            Predicted regression values.
        """
        X_tab = self._panel_to_tabular(X)
        return self.tabicl_regressor_.predict(X_tab)

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
