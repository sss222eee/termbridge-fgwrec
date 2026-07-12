"""Public entry point for the final TermBridge-FGWRec experiment."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()

