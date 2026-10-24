"""Filesystem/installed project loading with explicit project namespaces."""
__all__ = ["ProjectLoader", "ProjectPluginResolver"]


def __getattr__(name):
    if name in __all__:
        from inferops.adapters.outbound.projects import loader
        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
