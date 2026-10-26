"""Lada engine: everything that touches the upstream lada package directly.

Kept import-light on purpose: the submodules (adapter, io_hooks, cancel) import torch / lada.
"""
