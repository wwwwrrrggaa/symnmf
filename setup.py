from setuptools import setup, Extension

c_module_extension = Extension(
    'symnmfmodule',  # The name of the module importable in Python
    sources=['symnmfmodule.c', 'symnmf.c']  # Both the wrapper and the logic file are required
)

setup(
    name='symnmfmodule',
    version='1.0',
    description='C extension for Symmetric Non-negative Matrix Factorization optimization',
    ext_modules=[c_module_extension]
)
