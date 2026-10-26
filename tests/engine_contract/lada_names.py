"""The Lada names the contract tests drive, from their canonical homes (oh_player no longer re-exports them).

`namespace()` imports oh_player first (it points LADA_MODEL_WEIGHTS_DIR at the checkout before lada is imported) and
returns an object whose attributes resolve in oh_player, then the engine's adapter / io_hooks / cancel modules."""


class _Namespace:
    def __init__(self, modules):
        self._modules = modules

    def __getattr__(self, name):
        for m in self._modules:
            if hasattr(m, name):
                return getattr(m, name)
        raise AttributeError(name)


def namespace():
    from oh_streamer.player import oh_player
    from oh_streamer.engines.lada import adapter, cancel, io_hooks
    return _Namespace((oh_player, adapter, io_hooks, cancel))
