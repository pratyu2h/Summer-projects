# Multi-Label Chest X-Ray Classifier

Upgrade of the original binary (Pneumonia vs Normal) ResNet-50 classifier
to multi-label classification across all 14 NIH ChestX-ray14 findings,
with Grad-CAM interpretability and a Gemini-based clinical decision
support layer.

## What changed from v1

The original `pneumonia classifier.ipynb` (kept at the repo root) was a
binary classifier on the Kaggle Chest X-Ray Images dataset, with a few
issues: a near-empty validation split, no validation loop during
training, and hardcoded local Windows paths. This version:

| | v1 | v2 (this folder) |
|---|---|---|
| Task | Binary (Pneumonia / Normal) | Multi-label, 14 findings |
| Dataset | Kaggle Chest X-Ray (5,863 img, 2 classes) | NIH ChestX-ray14 (up to 112,120 img, 14 classes) |
| Loss | `CrossEntropyLoss` | Multi-label focal loss, class-frequency-weighted |
| Validation | Broken (near-empty split, no loop) | Proper patient-level split + per-epoch validation |
| Metric | Accuracy | Per-class + mean AUROC (matches CheXNet paper's evaluation) |
| Interpretability | None | Grad-CAM heatmaps |
| Clinical layer | None | Gemini API turns probabilities into a readable note |

## Architecture

```
Image (224x224x3)
   -> ResNet-50 backbone (ImageNet pretrained, progressive unfreezing)
   -> Dropout + Linear(2048 -> 14)
   -> 14 raw logits (one per finding, independent sigmoids -- multi-label, not softmax)
```

Training: focal loss (not plain BCE) because class prevalence ranges from
~0.2% (Hernia) to ~18% (Infiltration); per-class alpha is derived from
training-set frequency (`losses.compute_alpha_from_freq`) rather than
picked by hand. Backbone starts frozen (train just the head), then
unfreezes `layer3` onward partway through training with a 10x lower LR --
standard transfer-learning recipe that avoids destroying pretrained
features early when gradients are still noisy.

## Files

- `data.py` — `NIHChestXrayDataset`: parses the pipe-separated multi-label
  CSV into multi-hot vectors, patient-level train/val split.
- `model.py` — `MultiLabelCXRModel`: ResNet-50 + 14-way head, progressive
  unfreezing helper.
- `losses.py` — `MultiLabelFocalLoss` + frequency-based alpha weighting.
- `gradcam.py` — Grad-CAM implementation + heatmap overlay for
  visualizing which image regions drove a given finding's prediction.
- `gemini_utils.py` — builds a clinical-support prompt from the model's
  probabilities and calls the Gemini API. Explicitly framed as decision
  support, not diagnosis.
- `train.py` — training loop, per-class/mean AUROC, checkpointing on best
  validation AUROC.
- `pneumonia_classifier_v2.ipynb` — **run this one.** End-to-end Colab
  notebook: downloads data via Kaggle API, trains, evaluates, visualizes
  Grad-CAM, calls Gemini.

## Running it

This needs a GPU and internet access to Kaggle + Gemini, so it's built
for **Google Colab**, not local/offline execution:

1. Open `pneumonia_classifier_v2.ipynb` in Colab (Runtime → T4 GPU).
2. Have a Kaggle API token (`kaggle.json`) ready — free from
   kaggle.com/account.
3. Have a Gemini API key ready (only needed for the last section) —
   store it in Colab Secrets as `GEMINI_API_KEY`, don't paste it in a cell.
4. Run all cells top to bottom.

The notebook defaults to the `nih-chest-xrays/sample` dataset (5,606
images, ~1.2GB) so a full run finishes in ~10-15 minutes on a T4. Swap
to `nih-chest-xrays/data` (full 112,120 images, ~42GB) for numbers
comparable to the published CheXNet results — budget a few hours.

## Local development

`data.py`, `model.py`, `losses.py`, `gradcam.py`, `train.py` all have
`if __name__ == "__main__":` blocks that sanity-check their own logic on
synthetic data (no dataset download needed), useful if you're editing
this locally before running the real thing in Colab:

```bash
pip install torch torchvision pandas pillow scikit-learn matplotlib
python losses.py   # focal loss forward/backward
python model.py    # model forward shape + unfreeze logic
python data.py      # multi-label CSV parsing on a synthetic mini-dataset
python gradcam.py   # heatmap shape/range
```

## Notes / limitations

- AUROC on the sample subset will be noticeably lower than on the full
  dataset, especially for rare classes (Hernia, Fibrosis) — there just
  aren't enough positive examples in 5,606 images. Worth stating this
  explicitly if you cite results on a resume.
- The Gemini layer reasons over the CNN's *output probabilities*, not the
  raw image — it's a communication/reasoning layer on top of the vision
  model's decision, not a second opinion from a model that "looked at"
  the X-ray. Worth being precise about this distinction if asked about it
  in an interview.
- Not a diagnostic tool. The Gemini prompt and any UI built on top of
  this should keep saying so.
