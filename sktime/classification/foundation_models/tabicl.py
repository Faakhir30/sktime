"""TabICL tabular foundation model time series classifier."""

__author__ = ["fkiraly"]
__all__ = ["TabICLClassifier"]


from sktime.classification.base import BaseClassifier


class TabICLClassifier(BaseClassifier):
    r"""TabICL tabular foundation model time series classifier.

    Direct interface to ``TabICLClassifier`` from the ``tabicl`` package [1]_.

    TabICL [2]_ is a tabular in-context learning foundation model that performs
    classification via a single forward pass through a pre-trained transformer.
    It stores training data in-context and makes predictions at inference time.

    To use for time series classification, each time series instance is flattened
    to a 1D feature vector of length ``n_dims * n_timepoints``, which is then
    passed to the TabICL tabular model. For multivariate time series, dimensions
    are concatenated along the feature axis.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members. More members improve accuracy but increase
        runtime linearly.
    norm_methods : list of str or None, default=None
        Normalization methods to try. If None, defaults to TabICL built-in choices.
    feat_shuffle_method : str, default="latin"
        Feature permutation strategy for ensemble diversity.
    class_shuffle_method : str, default="shift"
        Class permutation strategy for ensemble diversity.
    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping.
    softmax_temperature : float, default=0.9
        Temperature to control prediction confidence.
    average_logits : bool, default=True
        If True, average logits across ensemble. If False, average probabilities.
    support_many_classes : bool, default=True
        If True, automatically handle >10 classes.
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
        (currently ``"tabicl-classifier-v2-20260212.ckpt"``).
        Available options:
        - "tabicl-classifier-v2-20260212.ckpt" (default): TabICLv2 [2]_
        - "tabicl-classifier-v1.1-20250506.ckpt": TabICLv1.1 (No Paper)
        - "tabicl-classifier-v1-20250208.ckpt": TabICLv2 [3]_
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
        Passed directly to the underlying ``TabICLClassifier``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during fit.
    n_classes_ : int
        Number of classes.
    tabicl_classifier_ : tabicl.TabICLClassifier
        The fitted underlying TabICL classifier.

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
    .. [3] @inproceedings{qu2025tabicl,
            title={Tab{ICL}: {A} Tabular Foundation Model for In-Context
            Learning on Large Data},
            author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux,
            Ga{\"e}l and Le Morvan, Marine},
            booktitle={International Conference on Machine Learning},
            year={2025}
            }

    Examples
    --------
    >>> from sktime.classification.foundation_models.tabicl import TabICLClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    >>> X_test, _ = load_unit_test(split="test", return_type="numpy3d")
    >>> clf = TabICLClassifier()  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    TabICLClassifier(...)  # doctest: +SKIP
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
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
        "capability:predict_proba": True,
        "capability:unequal_length": False,
        # testing flags
        # -------------
        "tests:vm": True,
    }

    def __init__(
        self,
        n_estimators=8,
        norm_methods=None,
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4.0,
        softmax_temperature=0.9,
        average_logits=True,
        support_many_classes=True,
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
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
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
        """Collect kwargs for TabICLClassifier instantiation.

        Returns
        -------
        kwargs : dict
            Keyword arguments for ``tabicl.TabICLClassifier``.
        """
        kwargs = {
            "n_estimators": self.n_estimators,
            "feat_shuffle_method": self.feat_shuffle_method,
            "class_shuffle_method": self.class_shuffle_method,
            "outlier_threshold": self.outlier_threshold,
            "softmax_temperature": self.softmax_temperature,
            "average_logits": self.average_logits,
            "support_many_classes": self.support_many_classes,
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
        """Fit the classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)
            Training time series panel data.
        y : np.ndarray of shape (n_instances,)
            Training class labels.

        Returns
        -------
        self : reference to self.
        """
        from tabicl import TabICLClassifier as _TabICLClassifier

        X_tab = self._panel_to_tabular(X)
        self.tabicl_classifier_ = _TabICLClassifier(**self._get_tabicl_kwargs())
        self.tabicl_classifier_.fit(X_tab, y)
        return self

    def _predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)
            Test time series panel data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_instances,)
            Predicted class labels.
        """
        X_tab = self._panel_to_tabular(X)
        return self.tabicl_classifier_.predict(X_tab)

    def _predict_proba(self, X):
        """Predict class probability estimates for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_dims, n_timepoints)
            Test time series panel data.

        Returns
        -------
        proba : np.ndarray of shape (n_instances, n_classes)
            Predicted class probabilities. Columns are in the order of
            ``self.classes_``.
        """
        X_tab = self._panel_to_tabular(X)
        return self.tabicl_classifier_.predict_proba(X_tab)

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
