"""Project-wide constants and paths.

All paths are derived relative to this file so the project can be run from any
working directory, and on any machine, without hardcoded absolute paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_YAML = DATA_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
RUNS_DIR = PROJECT_ROOT / "runs"

FLAGSHIP_MODEL_PATH = MODELS_DIR / "best.pt"
LIGHTWEIGHT_MODEL_PATH = MODELS_DIR / "best_lite.pt"

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

# Classes whose presence signals a PPE *violation* rather than compliant equipment
# or a neutral object -- used by the demo to highlight violations distinctly.
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

RANDOM_STATE = 42

DATASET_SOURCE_CITATION = (
    "Roboflow Universe: roboflow-universe-projects/construction-site-safety (v28), "
    "CC BY 4.0. Image/label files obtained via a working GitHub mirror "
    "(github.com/hoanganhtuoi125-gif/Construction-Site-Safety-PPE-Detection) after the "
    "originally specified mirror (github.com/snehilsanyal/Construction-Site-Safety-PPE-"
    "Detection) was found to not contain the actual image data -- see BUILD_LOG.md."
)
