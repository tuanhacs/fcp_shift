from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for package in [
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openml", "sanssouci"
    ]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
