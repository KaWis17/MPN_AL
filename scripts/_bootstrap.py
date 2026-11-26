"""Make ``mpn_al`` importable when a script runs from a plain checkout.

After ``pip install -e .`` this does nothing. Without an install it registers ``src/`` as the
``mpn_al`` package, which is the same mapping ``pyproject.toml`` declares. Scripts import this
module first: ``import _bootstrap  # noqa: F401``.
"""

import importlib.util
import sys
from pathlib import Path


def ensure_importable() -> None:
    if "mpn_al" in sys.modules or importlib.util.find_spec("mpn_al") is not None:
        return
    source = Path(__file__).resolve().parents[1] / "src"
    spec = importlib.util.spec_from_file_location(
        "mpn_al", source / "__init__.py", submodule_search_locations=[str(source)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mpn_al"] = module
    spec.loader.exec_module(module)


ensure_importable()
