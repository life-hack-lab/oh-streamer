"""The Lada names the dev benchmarks drive (FrameRestorer, ModelFiles, ...), from their canonical homes.

oh_player no longer re-exports them (engine-decoupling phase 3); these scripts measure Lada itself, so they may
reach into the Lada engine package. Importing this imports oh_player first (it points LADA_MODEL_WEIGHTS_DIR at the
checkout before lada is imported)."""


class _Namespace:
    def __init__(self, modules):
        self._modules = modules

    def __getattr__(self, name):
        for m in self._modules:
            if hasattr(m, name):
                return getattr(m, name)
        raise AttributeError(name)


def load():
    from oh_streamer.player import oh_player
    from oh_streamer.engines.lada import adapter, cancel, io_hooks
    return _Namespace((oh_player, adapter, io_hooks, cancel))
