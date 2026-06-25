#!/usr/bin/env python3
"""Clear execution state from the checked-in Strategy 2 notebook.

The notebook is maintained directly because it contains the current full
41-tile workflow. This helper prevents stale generated 24-tile notebook content
from overwriting the maintained notebook.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "strategy2.ipynb"


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"cleared outputs: {NOTEBOOK}")


if __name__ == "__main__":
    main()
