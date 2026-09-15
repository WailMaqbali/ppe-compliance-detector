# Summary

Built autonomously end to end. Repo is live and pushed:
**https://github.com/WailMaqbali/ppe-compliance-detector**

## What got built

A complete PPE compliance detection project on the Roboflow Universe
construction-site-safety dataset (2,801 images, 10 classes, 3.4% of the failure classes
are the safety-critical violation types):

- EDA (`notebooks/01_eda.ipynb`, executed with real outputs, plus `src/eda.py`) covering
  class distribution by split, annotation density, image size distribution, and sample
  annotated images -- including a real, non-obvious finding: a subset of the dataset's
  images are pre-tiled 2x2 photo mosaics baked in by Roboflow at export time.
- Two trained YOLOv8 models, both fine-tuned the full 200 epochs (patience-based early
  stopping never triggered for either): a **YOLOv8s flagship** (1.637 hours) and a
  **YOLOv8n lightweight** model (1.0 hour) built specifically for the live-webcam path,
  per the cross-device requirement (this machine trains on CUDA; the webcam code is meant
  to run on a MacBook Air M1's MPS backend tomorrow).
- Full evaluation for both models: precision/recall/mAP50/mAP50-95 overall and per-class,
  confusion matrices, PR curves, and real sample-detection figures, all saved to
  `figures/` (not left in the gitignored `runs/` folder).
- A working FastAPI-style Gradio demo (`src/app.py`) with image upload, video upload, and
  webcam tabs, plus a standalone OpenCV live-webcam script (`src/webcam_demo.py`). Both
  auto-detect CUDA / Apple Silicon MPS / CPU at runtime. Manually verified end-to-end on
  this machine: image annotation, video annotation (built and ran a synthetic test clip),
  and correct violation flagging, all against the real trained flagship model.
- 33 pytest tests, all real assertions, covering dataset integrity (full label-file
  validation, not just a sample), config/class-name sanity checks, detection-summary and
  violation-flag logic, Gradio app construction, and real inference against the trained
  model (shape checks, confidence-threshold monotonicity, class-name validity).
- README.md, MODEL_CARD.md, BUILD_LOG.md, this SUMMARY.md.
- 10 incremental git commits, pushed to a public GitHub repo, attribution-free throughout
  (verified via `git log` grep after every batch of commits).

## Headline real results

Held-out **test** split (82 images, 760 ground-truth boxes, never used for training or
validation-time model selection):

| Model | Params | GFLOPs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **YOLOv8s (flagship)** | 11.1M | 28.5 | 0.924 | 0.776 | **0.832** | 0.545 |
| YOLOv8n (lite, webcam) | 3.0M | 8.1 | 0.894 | 0.721 | 0.776 | 0.480 |

Two things worth knowing going in:

1. **`NO-Hardhat` recall is the weakest of the three violation classes** on both models
   (0.624 flagship / 0.585 lite) -- roughly 1 in 3 real missing-hardhat instances were
   missed by the flagship model in this test set. This is the single most important number
   in this project given the safety framing, and it's called out explicitly (not buried)
   in both README.md and MODEL_CARD.md.
2. **`Safety Cone` is the single weakest class overall** (recall 0.543 flagship / 0.439
   lite) -- less safety-critical, but the confusion matrix shows it's mostly false
   positives against background (the model over-triggers on cone-like shapes), not
   confusion with other classes. In fact, there is essentially **zero cross-class
   confusion** anywhere in either model -- every real error is a missed detection or a
   background false positive, never mislabeling one PPE class as another.

## An important process note: a real blocker, handled

The dataset mirror named in the original task turned out to not actually contain the
image/label data in its git history (only documentation and a couple of sample images).
Rather than stop, I verified (via the GitHub API, before cloning anything) that a
different public mirror of the same underlying Roboflow dataset had the real files --
matching split counts, matching class names, real JPEG blob sizes, well-formed label
lines -- and used that instead. Full details, including exactly what was checked and why,
are in `BUILD_LOG.md`. I also hit and fixed a real Ultralytics API quirk along the way:
the confusion matrix silently stays all-zero unless `val()` is called with `plots=True` in
this version -- also documented in `BUILD_LOG.md` and the `src/evaluate.py` code comment.

## What to look at first

1. `README.md` -- 60-second overview, results tables, embedded figures for both models.
2. `figures/flagship_pr_curve.png` and `figures/flagship_confusion_matrix.png` -- the
   clearest single view of where the model is strong (Hardhat, Safety Vest, Person) vs.
   weak (Safety Cone, NO-Hardhat).
3. `figures/flagship_sample_detections.png` -- real detections on real test images,
   including a genuine NO-Safety Vest + NO-Mask violation catch.
4. `MODEL_CARD.md` -- known limitations section, especially the NO-Hardhat discussion.
5. `src/app.py` (`python -m src.app`) to try the Gradio demo; `src/webcam_demo.py` for the
   standalone live-webcam script (untested with an actual physical camera in this
   headless session -- see below).

## What's still rough

- **The webcam path has not been run against a real physical camera.** This development
  session has no camera attached, so `src/webcam_demo.py` and the Gradio webcam tab are
  verified by code review and by testing the shared `annotate_image` function they both
  call against real images (which works correctly) -- but the actual `cv2.VideoCapture(0)`
  live-loop has not been visually confirmed. Please try it first when you're back, ideally
  on the MacBook Air per the original cross-device plan, and let me know if anything needs
  adjusting for MPS specifically.
- **GPU speed comparison between the two models was inconclusive** (measured lite as
  slightly *slower* than flagship in one noisy small-batch run on this desktop GPU) --
  not reported as a real result anywhere; the params/GFLOPs comparison is used instead as
  the honest basis for "lite is lighter." Real-time behavior on the M1's MPS backend is
  untested in this session.
- No CI (GitHub Actions) wired up yet to run `pytest` automatically on push.
- No hyperparameter search for either model -- both use sensible defaults with light
  manual choices (e.g. AutoBatch for batch size), so there's probably a bit more mAP on
  the table, especially for `NO-Hardhat` and `Safety Cone`.
- No object tracking across video frames -- every frame is detected independently.
- Full detail on every rough edge is in the README "What I'd improve" section and
  MODEL_CARD.md "Known limitations" -- nothing here is hidden or glossed over.

## One mid-build interruption

Partway through, you asked me to pause training and free the GPU for about an hour. By
that point both training runs had already finished on their own (flagship then lite, each
completing its full 200 epochs before your message arrived), so there was nothing to
actually pause -- I confirmed via `nvidia-smi` that no process was using the GPU and let
you know. All work after that point (finishing the README/MODEL_CARD writeup, this
summary) was CPU-only; I did not re-touch the GPU. If you'd like me to re-run any
GPU-touching verification (e.g. re-running the full pytest suite, live-testing the Gradio
app) now that you're done, just say so.
