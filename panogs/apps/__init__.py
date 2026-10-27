"""CLI and Application entry points for PanoGS."""

__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from panogs.apps.cli import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
