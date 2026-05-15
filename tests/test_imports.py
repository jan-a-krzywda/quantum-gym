import importlib


def test_top_level_shims_importable():
    # Import the shims (should not execute heavy ML code)
    importlib.import_module('ML')
    importlib.import_module('quantum_code')
    importlib.import_module('shadow_gym')
    # If import didn't raise, test passes
    assert True
