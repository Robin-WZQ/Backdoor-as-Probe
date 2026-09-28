"""Adversarial attack implementations and generation entry points."""

__all__ = ["ZeroShotCLIPClassifier", "cw_linf_attack", "pgd_attack"]


def __getattr__(name):
    if name in __all__:
        from . import methods

        return getattr(methods, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
