from setuptools import setup, Extension

spicemodule = Extension('spice',
                        sources=['spicemodule.c'],
                        include_dirs=['cspice/src/versn_c'],
                        libraries=[':cspice.a'],
                        library_dirs=['cspice/lib'])

setup(
    name='spice',
    version='1.0',
    description='Python interface for the SPICE Toolkit',
    packages=['.'],  # The package is in the current directory
    ext_modules=[spicemodule],
    package_data={'spice': ['spice.pyi']}  # Include .pyi for type hints
)