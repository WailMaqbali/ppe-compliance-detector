# Build Log

Running log of decisions, blockers, and anything that didn't go as planned while building
this project autonomously.

## Environment

- Windows machine, Python 3.13.3, NVIDIA GeForce RTX 4060 Ti (8.6 GB VRAM).
- `gh auth status` confirmed an authenticated GitHub CLI session (account: WailMaqbali).
- `git config user.name` / `user.email` confirmed set to the account's own identity
  before any commits were made.
- The project was scaffolded as a new sibling folder, `ppe-compliance-detector/`, next to
  the existing `predictive-maintenance` project (this is the second entry in a 4-project
  portfolio) rather than inside the shell's default working directory, to match the repo
  structure the task specified and keep the two projects fully separate.
- `pip install torch` pulled a CPU-only wheel by default in this environment (torch
  2.14.0+cpu). Reinstalled explicitly from the CUDA 12.6 wheel index
  (`--index-url https://download.pytorch.org/whl/cu126`) to get GPU support; confirmed
  `torch.cuda.is_available() == True` afterward, device = "NVIDIA GeForce RTX 4060 Ti".
  `requirements.txt` itself still lists a plain `torch>=2.2` with no hardcoded CUDA-only
  index URL, per the instruction to keep it portable to the Mac (MPS) target machine — the
  CUDA-specific install command is documented in the README as a training-machine-only step.

## Blocker: the specified dataset mirror does not actually contain the dataset

The task's clone command
(`git clone https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection.git`)
was run as instructed. The resulting repo's `data/` folder contains only `data.yaml`,
`ppe_data.yaml`, and two README files — **no `train/`, `valid/`, or `test/` image/label
folders exist anywhere in that repo's git history** (verified via `git ls-tree -r HEAD` and
`git log --all -- data/train`; zero results). The repo's own README documents a file
hierarchy that includes `data/train`, `data/valid`, `data/test` and describes 2,801 images
split 2,605/114/82 — matching the task's description exactly — but the actual image/label
files were evidently never committed to this particular mirror (no `.gitattributes` /
git-lfs pointers either; the data is just absent). `data.yaml`'s class names in this repo
*were* correct real words ("Hardhat", "Mask", etc.), so the "verify class names" check
alone would not have caught this — the problem was missing data, not corrupted labels.

**Resolution:** rather than stop with no dataset (which would block every remaining part of
this task), searched for another public mirror of the same Roboflow Universe dataset
(`roboflow-universe-projects/construction-site-safety`, version 28) that actually contains
the committed image/label files. Found
`github.com/hoanganhtuoi125-gif/Construction-Site-Safety-PPE-Detection`
(~389 MB, i.e. plausibly sized for ~2,800 images + labels + model weights, vs. the broken
mirror's few KB of text files). Verified *before* cloning, via the GitHub API (no local
clone needed for this check):
- Split counts via `git/trees` recursive listing: `train/images` = 2605, `valid/images` =
  114, `test/images` = 82 — an exact match to the task's stated split.
- `data.yaml` (at `Model-Training/Outputs/data.yaml` in that repo) has the same 10 class
  names in the same order as the specified dataset.
- A sample image blob was 61,784 bytes (a real JPEG, not a ~130-byte git-lfs pointer stub).
- A sample label file contained well-formed YOLO-format lines (`class x_center y_center
  width height`, all in [0, 1], class indices in range).

Cloned only the relevant subtree with `git clone --filter=blob:none --sparse` +
`git sparse-checkout set Model-Training/Dataset` (avoids pulling that repo's own
`yolov8s.pt`/`yolo11n.pt` weights, training notebook, or prior run outputs), copied
`train/`, `valid/`, `test/` into this project's `data/`, hand-wrote `data/data.yaml` with
the verified class list and project-relative paths, and deleted the temporary clone.

**Full-dataset integrity check** (all 2,605 + 114 + 82 label files, not just a sample):
every image has exactly one matching label file and vice versa (zero missing, zero
orphaned), every label line parses as 5 fields with a class index in `[0, 9]` (zero
malformed lines), and the per-class box counts across the whole dataset are:

| Class | Boxes |
|---|---|
| Hardhat | 3,334 |
| Mask | 1,700 |
| NO-Hardhat | 2,427 |
| NO-Mask | 3,250 |
| NO-Safety Vest | 4,158 |
| Person | 9,872 |
| Safety Cone | 3,502 |
| Safety Vest | 3,135 |
| machinery | 5,346 |
| vehicle | 1,628 |

38,352 total boxes across 2,801 images. Some class imbalance (Mask and vehicle are the
least represented at ~1,600-1,700 boxes vs. Person at ~9,900), noted for the evaluation and
MODEL_CARD limitations sections, but no class is a data-integrity red flag.

Dataset citation note: same underlying Roboflow Universe dataset
(`roboflow-universe-projects/construction-site-safety`, v28) as originally specified; only
the GitHub mirror used to obtain the files differs from the one named in the task. Both
facts are recorded in the README citation.
