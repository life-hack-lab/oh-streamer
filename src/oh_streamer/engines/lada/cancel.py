# SPDX-License-Identifier: AGPL-3.0-only
# Adds run-time hooks to lada's restoration model (upstream lada, AGPL-3.0). Modified 2026-10-26; see THIRD_PARTY_NOTICES.md, section
# "Modifications to upstream software".
"""Abandoning an in-progress clip restoration (fast stop) for the Lada engine.

Moved verbatim from player/oh_player.py (engine-decoupling phase 2); behavior unchanged.
"""

import threading

import torch

# ---------------------------------------------------------------------------
# Abandoning an in-progress clip restoration (fast stop)
# ---------------------------------------------------------------------------
# FrameRestorer.stop() joins the clip restoration worker, which may be in the middle of restoring a long clip
# (seconds). Each restorer gets its own proxy of the shared restoration model with a cancel event; while the proxy's
# restore() runs, the worker thread's thread-local points at that event, and pre-forward hooks on the model's
# (non-leaf) submodules abort the forward pass when it is set. The proxy then returns the unrestored frames, which the
# stopping pipeline discards. Hooks of other threads / restorers see no event and do nothing.

class _RestoreCancelled(Exception):
    pass


_cancel_tls = threading.local()


def _cancel_check_hook(_module, _inputs):
    ev = getattr(_cancel_tls, "event", None)
    if ev is not None and ev.is_set():
        raise _RestoreCancelled()


def install_cancel_hooks(res_model) -> int:
    """Register the cancel check on the non-leaf submodules of the restoration network (once per model load)."""
    n = 0
    for v in vars(res_model).values():
        if isinstance(v, torch.nn.Module):
            for m in v.modules():
                if next(m.children(), None) is not None:   # leaves (convs, activations) are called far too often
                    m.register_forward_pre_hook(_cancel_check_hook)
                    n += 1
    return n


def cancellable_restoration_model(res_model):
    """Per-restorer proxy (an instance of a subclass of the model wrapper's class, so Lada's isinstance checks pass)
    sharing all state with res_model, plus .cancel_event."""
    base = type(res_model)

    def restore(self, video, *a, **kw):
        _cancel_tls.event = self.cancel_event
        try:
            return base.restore(self, video, *a, **kw)
        except _RestoreCancelled:
            return list(video)
        finally:
            _cancel_tls.event = None

    cls = type(f"Cancellable{base.__name__}", (base,), {"restore": restore})
    proxy = object.__new__(cls)
    proxy.__dict__.update(res_model.__dict__)
    proxy.cancel_event = threading.Event()
    return proxy

