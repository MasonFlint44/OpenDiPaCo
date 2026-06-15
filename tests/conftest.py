import sys
from pathlib import Path

# Make the package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Make the validate_* harnesses (examples/) importable so their scenarios can be
# reused as integration tests (e.g. tests/test_churn.py reuses validate_churn).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
