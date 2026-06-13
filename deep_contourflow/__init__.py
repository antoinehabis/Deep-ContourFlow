"""Deep ContourFlow (DCF) — training-free active contours powered by deep features.

Two segmentation regimes are exposed:

- :class:`UnsupervisedDCF` — evolve a contour to maximize the feature contrast
  between the inside and the outside of the curve (no labels needed).
- :class:`OneShotDCF` — segment a query image from a single annotated support
  example (support image + mask).

Example
-------
>>> from deep_contourflow import UnsupervisedDCF
>>> dcf = UnsupervisedDCF(model="vgg16", n_epochs=100)
>>> contours, loss_history, final_contour = dcf.predict(image, contour_init)
"""

from .oneshot import DCF as OneShotDCF
from .unsupervised import DCF as UnsupervisedDCF

__all__ = ["UnsupervisedDCF", "OneShotDCF"]
