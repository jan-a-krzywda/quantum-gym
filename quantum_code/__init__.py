"""Top-level shim for backwards compatibility.

Re-exports the implementation moved to `calibration-gym/quantum_code` so
imports like `import quantum_code` continue to work.
"""
import importlib
import pkgutil
__path__.insert(0, importlib.import_module('os').path.join(importlib.import_module('os').path.dirname(__file__), '..', 'calibration-gym', 'quantum_code'))

__all__ = [name for _, name, _ in pkgutil.iter_modules(__path__)]
