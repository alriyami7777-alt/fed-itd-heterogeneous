"""Single path setting for the whole package. Set FITD_ROOT to the folder that holds data/ and results/
(default: the repository root, one level above src/)."""
import os

ROOT = os.environ.get("FITD_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "{}_user_day_dev.parquet")
RESULTS = os.path.join(ROOT, "results")
