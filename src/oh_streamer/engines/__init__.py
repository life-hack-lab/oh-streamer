"""Restoration engines: the interface the player/queue program against (base.py) and the engines behind it.

Importing this package never imports torch or lada; get_engine() imports the requested engine lazily.

Which engine runs, highest priority first: the --engine flag (set_default_engine) or the OH_STREAMER_ENGINE environment
variable (both developer overrides), then the `engine` setting from the settings file / queue job
(set_configured_engine), then "lada".
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_ENGINE = "lada"
ENGINE_NAMES = ("lada", "passthrough")      # the built-in engines (argparse choices); engine_names() also lists registered ones


@dataclass(frozen=True)
class _Entry:
    factory: Callable                       # () -> engine instance (imports the engine lazily)
    user_visible: bool = True               # False: developer / test engine, never offered in a UI
    label: str = ""                         # UI name
    catalog: Optional[Callable] = None      # () -> ModelCatalog without loading the engine (no torch); None: get_engine(name).list_models()


_REGISTRY: dict = {}
_ENGINES: dict = {}
_selected: list = []      # the engine the process was told to use (set_default_engine); else OH_STREAMER_ENGINE
_configured: list = []    # the engine named by the settings (set_configured_engine)
_logged: set = set()


def register_engine(name: str, factory: Callable, *, user_visible: bool = True, label: str = "",
                    catalog: Optional[Callable] = None) -> None:
    _REGISTRY[name] = _Entry(factory, user_visible, label or name, catalog)
    _ENGINES.pop(name, None)


def unregister_engine(name: str) -> None:
    _REGISTRY.pop(name, None)
    _ENGINES.pop(name, None)
    if _selected and _selected[0] == name:
        _selected.clear()
    if _configured and _configured[0] == name:
        _configured.clear()


def engine_names() -> tuple:
    return tuple(_REGISTRY)


def visible_engine_names() -> tuple:
    """The engines a UI may offer. A selector is only worth showing when there is more than one."""
    return tuple(n for n, e in _REGISTRY.items() if e.user_visible)


def engine_label(name: str) -> str:
    e = _REGISTRY.get(name)
    return e.label if e else name


def _unknown(name) -> ValueError:
    return ValueError(f"unknown restoration engine: {name!r} (available: {', '.join(_REGISTRY)})")


def explicit_engine_name() -> Optional[str]:
    """The developer override: --engine (set_default_engine), else the OH_STREAMER_ENGINE environment variable, else None."""
    name = _selected[0] if _selected else os.environ.get("OH_STREAMER_ENGINE", "").strip() or None
    if name is not None and name not in _REGISTRY:
        raise _unknown(name)
    return name


def default_engine_name() -> str:
    """The engine the callers use: --engine, else OH_STREAMER_ENGINE, else the settings' engine (set_configured_engine), else lada."""
    return explicit_engine_name() or (_configured[0] if _configured else DEFAULT_ENGINE)


def set_default_engine(name: str) -> None:
    if name not in _REGISTRY:
        raise _unknown(name)
    _selected[:] = [name]


def resolve_engine_name(name, log: Optional[Callable] = None) -> str:
    """The engine a settings value asks for; a missing, unknown or unavailable one falls back to lada (never raises)."""
    if name is None or name == "":
        return DEFAULT_ENGINE
    if isinstance(name, str) and name in _REGISTRY:
        return name
    if log is not None and name not in _logged:
        _logged.add(name)
        log(f"restoration engine {name!r} is not available, using {DEFAULT_ENGINE!r}")
    return DEFAULT_ENGINE


def effective_engine_name(setting, log: Optional[Callable] = None) -> str:
    """--engine / OH_STREAMER_ENGINE if given, else the settings value (resolved, with the lada fallback)."""
    return explicit_engine_name() or resolve_engine_name(setting, log)


def set_configured_engine(name, log: Optional[Callable] = None) -> str:
    """Make the settings' engine the process default (unless --engine / OH_STREAMER_ENGINE override it). Returns the name in use."""
    resolved = resolve_engine_name(name, log)
    _configured[:] = [resolved]
    return resolved


def _create(name: str):
    return _REGISTRY[name].factory()


def get_engine(name: str | None = None, *, new: bool = False):
    """The process-wide engine registered under `name` (default: default_engine_name(); lazy import; ValueError for an
    unknown name). new=True builds a separate instance instead (tests); it is not remembered."""
    if name is None:
        name = default_engine_name()
    if name not in _REGISTRY:
        raise ValueError(f"unknown restoration engine: {name!r}")
    if new:
        return _create(name)
    if name not in _ENGINES:
        _ENGINES[name] = _create(name)
    return _ENGINES[name]


def model_catalog(name: str | None = None):
    """The models engine `name` (default: the one in use) offers, read without loading it when the engine can."""
    name = name or default_engine_name()
    if name not in _REGISTRY:
        raise _unknown(name)
    light = _REGISTRY[name].catalog
    return light() if light is not None else get_engine(name).list_models()


def _lada():
    from oh_streamer.engines.lada.engine import LadaEngine   # lazy: imports torch and lada
    return LadaEngine()


def _lada_catalog():
    from oh_streamer.engines.lada import catalog           # reads the weights folder; imports neither torch nor lada
    return catalog.installed_models()


def _passthrough():
    from oh_streamer.engines.passthrough.engine import PassthroughEngine   # lazy: imports torch and av
    return PassthroughEngine()


register_engine("lada", _lada, label="Lada", catalog=_lada_catalog)
register_engine("passthrough", _passthrough, user_visible=False, label="Passthrough")   # developer flag, restores nothing
