"""Classification foundation models."""

__all__ = [
    "MomentFMClassifier",
    "TabICLClassifier",
]

from sktime.classification.foundation_models.momentfm import MomentFMClassifier
from sktime.classification.foundation_models.tabicl import TabICLClassifier
