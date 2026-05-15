"""Top-level shim for backwards compatibility.

This package transparently re-exports the real implementation moved to
`calibration-gym/ML` so existing import paths like `import ML` keep working.
"""
import importlib
import pkgutil
__path__.insert(0, importlib.import_module('os').path.join(importlib.import_module('os').path.dirname(__file__), '..', 'calibration-gym', 'ML'))

# Ensure package is importable; expose submodules normally.
__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
