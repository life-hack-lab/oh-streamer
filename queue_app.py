"""Compatibility shim for oh_streamer.queue.queue_app."""
from pathlib import Path as _Path
import sys as _sys

_source = str((_Path(__file__).resolve().parent / "src").resolve())
if not _sys.path or _sys.path[0] != _source:
    _sys.path.insert(0, _source)

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module('oh_streamer.queue.queue_app', run_name="__main__", alter_sys=True)
else:
    from importlib import import_module as _import_module
    _module = _import_module('oh_streamer.queue.queue_app')
    _sys.modules[__name__] = _module
