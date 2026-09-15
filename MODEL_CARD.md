# Model Card: PPE Compliance Detector

Follows the general structure of Mitchell et al., "Model Cards for Model Reporting" (2019).

## Model description

- **Task:** Object detection -- localize and classify PPE-related objects and people in
  construction-site images/video: `Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
  Person, Safety Cone, Safety Vest, machinery, vehicle`.
- **Flagship model:** YOLOv8s, fine-tuned from COCO-pretrained weights on the PPE dataset
  (`models/best.pt`). Used by the Gradio demo's image/video tabs and reported in the
  Evaluation results below.
- **Lightweight model:** YOLOv8n, separately fine-tuned on the same dataset
  (`models/best_lite.pt`). Used only by the live-webcam path (`src/webcam_demo.py` and the
  Gradio app's webcam tab), where smooth real-time inference on modest hardware (e.g. a
  MacBook Air M1's MPS backend) matters more than the last bit of mAP -- see README.md.
- **Architecture:** Ultralytics YOLOv8, anchor-free, single-stage detector.

## Intended use

Built as a **portfolio and educational project** demonstrating a real-time PPE compliance
detection pipeline: EDA, GPU fine-tuning with early stopping, honest per-class evaluation,
and both an upload-based and live-webcam demo. It is **not a certified safety system**.
Before any real deployment on an actual construction site (e.g. as an automated compliance
alarm), it would need: substantially more and more diverse training data (this dataset's
2,801 images come from a specific public collection, not site-specific footage),
domain/safety-expert validation, false-negative-rate testing against the specific site
conditions (lighting, camera angle, PPE styles) it would run against, and a human-in-the-
loop review process rather than fully automated alarm triggering.

**Out of scope:** identity tracking / individual worker attribution, and any automated
enforcement action (e.g. access control) -- this model only detects and classifies visible
objects in a frame; it says nothing about who a person is or what should happen next.

## Training data

[Roboflow Universe: Construction Site Safety Image Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
(v28), CC BY 4.0, 2,801 images total, split train 2,605 / valid 114 / test 82
(pre-split by the dataset publisher, used as-is). Sourced via a working public GitHub
mirror after the originally-specified mirror was found to be missing the actual image
data -- see `BUILD_LOG.md`.

Box counts by class, across the full dataset (38,352 boxes total):

| Class | Boxes | % of all boxes |
|---|---|---|
| Person | 9,872 | 25.7% |
| machinery | 5,346 | 13.9% |
| NO-Safety Vest | 4,158 | 10.8% |
| Safety Cone | 3,502 | 9.1% |
| NO-Mask | 3,250 | 8.5% |
| Hardhat | 3,334 | 8.7% |
| Safety Vest | 3,135 | 8.2% |
| NO-Hardhat | 2,427 | 6.3% |
| Mask | 1,700 | 4.4% |
| vehicle | 1,628 | 4.2% |

`Person` and `machinery` dominate; `Mask` and `vehicle` have the least support. All three
violation classes (`NO-Hardhat`, `NO-Safety Vest`, `NO-Mask`) have reasonable box counts
(2,400-4,200 each), so the weak `NO-Hardhat` result below is not simply a training-data
volume problem the way, e.g., a sub-50-example class would be -- see Known limitations.

A meaningful subset of source images are pre-tiled 2x2 photo mosaics baked in by Roboflow
at export time (four unrelated site photos combined into one image, each quadrant
separately annotated) -- see `notebooks/01_eda.ipynb`. Kept in training (removing them
would discard real, correctly-labeled data), but flagged as a limitation below.

## Evaluation results

Held-out **test** split (82 images, 760 ground-truth boxes; never used for training or for
validation-time checkpoint selection). Exact output of `python -m src.evaluate --weights
models/best.pt --split test` (`models/flagship_metrics.json`), not rounded favorably.

**Overall: precision 0.924, recall 0.776, mAP50 0.832, mAP50-95 0.545.**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Hardhat | 0.996 | 0.909 | 0.944 | 0.672 |
| Mask | 0.958 | 0.750 | 0.817 | 0.564 |
| NO-Hardhat | 0.836 | 0.624 | 0.628 | 0.377 |
| NO-Mask | 0.858 | 0.810 | 0.879 | 0.478 |
| NO-Safety Vest | 0.972 | 0.779 | 0.834 | 0.584 |
| Person | 0.955 | 0.855 | 0.905 | 0.624 |
| Safety Cone | 0.879 | 0.543 | 0.587 | 0.272 |
| Safety Vest | 0.982 | 0.891 | 0.956 | 0.664 |
| machinery | 0.887 | 0.841 | 0.900 | 0.689 |
| vehicle | 0.911 | 0.756 | 0.866 | 0.529 |

Confusion matrix (`figures/flagship_confusion_matrix.png`) shows essentially **zero
cross-class confusion** among the ten PPE/object classes -- every off-diagonal error is
either a missed detection (true box, nothing predicted) or a false positive against
background (something predicted, no true box there). The model does not, for example,
confuse `Hardhat` with `NO-Hardhat`; when it's wrong, it's usually because it missed the
object entirely rather than mislabeling it.

**Lightweight YOLOv8n webcam model**, same test split: precision 0.894, recall 0.721,
mAP50 0.776, mAP50-95 0.480 -- a real but modest drop from the flagship (mAP50 0.832),
traded for 3.5x fewer parameters (3.0M vs 11.1M) and 3.5x fewer GFLOPs (8.1 vs 28.5), the
actual basis for using it on constrained hardware (see README "Flagship vs. lightweight
model"). Its weakest class is also `Safety Cone` (recall 0.439, mAP50 0.477) -- weaker
still than the flagship's own Safety Cone result -- and `NO-Hardhat` recall is 0.585,
also below the flagship's 0.624. Both models share the same weak spots; the lite model is
simply weaker everywhere, as expected for a much smaller backbone.

## Known limitations

- **`NO-Hardhat` recall is the weakest of the three violation classes: 0.624** (vs. 0.779
  for `NO-Safety Vest` and 0.810 for `NO-Mask`), with mAP50 of only 0.628. This is the
  most safety-relevant number in this model card: **roughly 1 in 3 real `NO-Hardhat`
  instances in this test set were missed.** This is not a training-data volume problem in
  the obvious sense -- `NO-Hardhat` has 2,427 boxes in training, more than `Mask` (1,700)
  or `vehicle` (1,628), both of which score higher on mAP50-95. A plausible explanation
  (not confirmed) is that a bare head is a visually smaller, less distinctive target than
  a brightly colored hardhat or vest, especially at the range/angle typical in this
  dataset's photos -- worth targeted investigation before any real deployment.
- **`Safety Cone` is the single weakest class overall** (recall 0.543, mAP50 0.587,
  mAP50-95 0.272). Less safety-critical than `NO-Hardhat`, but the confusion matrix shows
  the largest false-positive count of any class (37 background objects predicted as Safety
  Cone), suggesting the model over-triggers on cone-like shapes/colors.
- **Pre-tiled mosaic training images** (see Training data) mean part of the training
  distribution has an artificial four-panel layout that will not appear in real deployment
  video or photos. Not confirmed to hurt performance, but not controlled for either.
- **Small dataset by modern object-detection standards.** 2,605 training images is enough
  to fine-tune a COCO-pretrained backbone to a reasonable mAP (this is transfer learning,
  not training from scratch), but is small relative to the visual diversity of real
  construction sites worldwide -- see "What I'd improve" in README.md.
- **The webcam path uses the weaker lightweight model.** Live/webcam inference runs the
  YOLOv8n checkpoint specifically because it's fast enough for real-time use, but its
  `NO-Hardhat` recall (0.585) and `Safety Cone` recall (0.439) are both worse than the
  flagship's already-weak numbers for those classes. A live compliance-monitoring demo is
  therefore the *least* reliable mode of this project for exactly the violation class that
  matters most -- worth stating plainly rather than only in a table.
- **No cross-class confusion, but real miss rates.** As noted in Evaluation results, the
  model rarely mislabels one PPE class as another -- its errors are almost entirely missed
  detections or background false positives, not e.g. confusing a hardhat for its absence.
  That's a meaningfully different (and arguably safer-to-debug) failure mode than
  systematic misclassification, but it doesn't change the raw miss rates above.
- **Single-frame detection only** -- no object tracking across video frames, so the same
  worker can be counted as a fresh violation on every frame of a video, and a single
  missed frame doesn't automatically get "caught" on the next one the way a tracker would.

## Ethical / safety considerations

This is a compliance-monitoring aid, not a safety interlock. The two error types have
asymmetric real-world consequences:

- **False negative** (model fails to flag a `NO-Hardhat` / `NO-Mask` / `NO-Safety Vest`
  violation that is actually present): the practical implication is a **missed PPE
  violation** -- a worker without required protective equipment continues undetected,
  which in a real deployment could mean a preventable injury goes unprevented. This is the
  most safety-relevant failure mode of this model: overall recall on the test set is 0.776,
  and `NO-Hardhat` recall specifically is only 0.624 -- roughly 1 in 3 missed -- quantified
  fully above (see Evaluation results / Known limitations) rather than glossed over.
- **False positive** (model flags a violation that isn't real, or misclassifies a
  compliant worker): costs unnecessary intervention time and, at scale, alert fatigue that
  could cause real alerts to be deprioritized or ignored -- the same alert-fatigue dynamic
  noted in this author's other portfolio projects.

Given this asymmetry, and the limitations above, this model should never be the sole basis
for a real safety decision or automated enforcement action. It is a monitoring aid, and any
operational use would need the additional validation work noted in "Intended use," plus
human review of flagged (and, ideally, periodic audit of unflagged) footage.
