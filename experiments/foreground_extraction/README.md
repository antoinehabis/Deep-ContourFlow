# Foreground extraction with DCF

Runs the **unsupervised** Deep ContourFlow algorithm as a foreground/background
segmenter over the three benchmark datasets (`ecssd`, `msra_b`, `cub`) and scores
the predictions against the ground-truth masks.

For the datasets themselves and the headline numbers, see
[**Benchmark** in the root README](../../README.md#-benchmark--reproducing-the-results).

## Layout

```
conf/config.yaml  # Hydra config: DCF hyperparameters + data settings (edit this)
dataset.py        # ForegroundDataset + DataLoader + frame crop/remap + contour helpers
metrics.py        # IoU, Dice, pixel accuracy + running-mean accumulator
eval.py           # Hydra entry point: run DCF over batches, aggregate metrics
results/          # one timestamped folder per run, plus latest.json
```

Datasets are read from `datasets/formatted/<name>/<image_id>/{image.jpg, mask.jpg}`,
produced by `scripts/download_datasets.py` + `scripts/format_datasets.py`
(`make datasets`).

## Usage

Configuration is managed with [Hydra](https://hydra.cc). Run from the **repo
root** (so `deep_contourflow` and `experiments` both import) and override any
field with `key=value`:

```bash
# Sanity check: 4 ECSSD images, 15 evolution steps (~1 min)
python -m experiments.foreground_extraction.eval 'datasets=[ecssd]' data.max_samples=4 dcf.n_epochs=15

# Fast benchmark: a seeded 200-image subset of each dataset (the README table)
python -m experiments.foreground_extraction.eval data.max_samples=200

# Full benchmark on all 17 788 images (config defaults)
python -m experiments.foreground_extraction.eval

# Multirun sweep over a hyperparameter (-m)
python -m experiments.foreground_extraction.eval -m dcf.learning_rate=0.01,0.05,0.1
```

`conf/config.yaml` pins **every** DCF hyperparameter explicitly rather than
inheriting library defaults, so a change to a default in `deep_contourflow`
cannot silently move the published numbers.

`data.max_samples` draws a **seeded random subset** (`data.seed`, default 1234)
instead of truncating the sorted id list — the ids are sorted by class on CUB, so
taking the first N would evaluate a single bird species.

Each run writes `results.json` into its own timestamped folder under `results/`,
and refreshes `results/latest.json`.

## How it works

1. Each image is resized to a square `data.image_size` and normalized to `[0, 1]`.
2. **Frame removal** (`data.remove_frame`, on by default): a solid border frame is
   detected and cropped, and the content is resized back to `image_size`. This
   stops DCF from locking onto the frame edge instead of the object inside it.
   Detection is conservative — a side is cropped only when its border is a
   near-solid color *and* gives way to the content through a sharp edge, so smooth
   skies and studio backdrops are left alone.
3. The contour is initialized as a circle (`init.shape`/`size`/`nb_nodes`) and
   evolved by `UnsupervisedDCF.predict` to maximize inside/outside feature
   contrast, stopping at the knee of the energy curve.
4. The final contour is rasterized to a binary mask with `cv2.fillPoly` (the
   same convention the package uses internally).
5. The mask is **remapped back** into the full `image_size` canvas at the crop
   location (zeros over the cropped-out frame band) so it aligns with the
   ground truth.
6. Metrics are computed per sample at `image_size` resolution, averaged within a
   dataset, then macro-averaged across datasets (equal weight per dataset, so the
   11 788-image CUB does not drown out the 1 000-image ECSSD).

## Programmatic use

```python
from experiments.foreground_extraction import load_cfg, evaluate

cfg = load_cfg()
cfg.data.max_samples = 25
print(evaluate(cfg, "ecssd")["metrics"])
```
