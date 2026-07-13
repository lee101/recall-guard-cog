from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _candidate_paths() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    major = sys.version_info.major
    minor = sys.version_info.minor
    tag = f"cpython-{major}{minor}"
    candidates = [
        root / f"hnswlib.{tag}-darwin.so",
        root / "build" / f"lib.macosx-15.0-arm64-cpython-{major}{minor}" / f"hnswlib.{tag}-darwin.so",
        root / "build" / f"lib.macosx-10.9-universal2-cpython-{major}{minor}" / f"hnswlib.{tag}-darwin.so",
    ]
    # App.nz addition: the upstream loader only searched macOS build names,
    # which makes a source checkout shadow a correctly installed Linux wheel.
    # Discover the ABI-tagged extension in the checkout, build tree, or active
    # site-packages without assuming a particular manylinux platform suffix.
    for base in [root, root / "build", *(Path(p) for p in sys.path if p)]:
        if base.exists():
            candidates.extend(base.glob(f"**/hnswlib.{tag}-*.so"))
    return candidates


for _path in _candidate_paths():
    if _path.exists():
        _spec = importlib.util.spec_from_file_location(__name__, str(_path))
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not create import spec for {_path}")
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[__name__] = _mod
        _spec.loader.exec_module(_mod)
        globals().update(_mod.__dict__)
        break
else:
    raise ImportError(
        f"Could not find local hnswlib extension for Python "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
