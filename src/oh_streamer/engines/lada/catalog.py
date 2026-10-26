"""The models the Lada engine can load, read from Lada's declarative registry and the weights folder without importing
lada or torch (the queue app lists them in a process that must stay light)."""

import ast
import importlib.util
import os
from pathlib import Path

from oh_streamer.engines import base
from oh_streamer.paths import upstream_root


def _registry_file() -> Path:
    """lada's ``__init__.py``: wherever the importable ``lada`` package lives (an installed wheel in site-packages, or an
    editable/source checkout on the path), else the legacy ``<root>/lada/lada`` checkout. ``find_spec`` of a top-level
    package only locates it and executes nothing (neither lada nor torch gets imported)."""
    try:
        spec = importlib.util.find_spec("lada")
        if spec is not None and spec.origin and Path(spec.origin).is_file():
            return Path(spec.origin)
    except (ImportError, ValueError):   # ValueError: a module in sys.modules without __spec__
        pass
    return upstream_root() / "lada" / "lada" / "__init__.py"


def installed_models() -> base.ModelCatalog:
    """The names LadaEngine.list_models() returns for the checked-out weights; empty when the registry can't be read."""
    root = Path(os.environ.get("LADA_MODEL_WEIGHTS_DIR", upstream_root() / "lada" / "model_weights"))
    result = {"detection_model": [], "restoration_model": []}
    try:
        tree = ast.parse(_registry_file().read_text(encoding="utf-8"))
        registry = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelFiles")
        files = list(root.iterdir()) if root.is_dir() else []
        for key, attr, prefix, suffix in (
            ("detection_model", "_WELL_KNOWN_DETECTION_MODELS", "lada_mosaic_detection_model_", ".pt"),
            ("restoration_model", "_WELL_KNOWN_RESTORATION_MODELS", "lada_mosaic_restoration_model_", ".pth"),
        ):
            declaration = next(n.value for n in registry.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == attr for t in n.targets))
            known = {}
            for model in declaration.elts:
                name = ast.literal_eval(model.args[0])
                parts = [ast.literal_eval(arg) for arg in model.args[2].args[1:]]
                known[root.joinpath(*parts)] = name
            names = [name for path, name in known.items() if path.is_file()]
            for path in sorted(files):
                if path in known or not path.is_file() or not path.name.startswith(prefix) or path.suffix != suffix:
                    continue
                name = path.stem[len(prefix):]
                if not name:
                    continue
                if key == "restoration_model":
                    if not name.startswith("deepmosaics") and "deepmosaics" in name:
                        name = "deepmosaics-" + name
                    elif not name.startswith("basicvsrpp"):
                        name = "basicvsrpp-" + name
                names.append(name)
            result[key] = names
    except (OSError, SyntaxError, ValueError, StopIteration, AttributeError, TypeError):
        pass
    return base.ModelCatalog(tuple(result["detection_model"]), tuple(result["restoration_model"]))
