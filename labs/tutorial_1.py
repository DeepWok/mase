"""Compatibility wrapper for legacy lab command.

Preferred command:
    uv run python docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py
"""

from pathlib import Path
import runpy


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py"
)


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
