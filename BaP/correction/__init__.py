"""Canonical single-start TTC escape plus probe repair."""

__all__ = ["correct_batch"]


def __getattr__(name):
    if name in __all__:
        from .current import correct_batch

        return correct_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
