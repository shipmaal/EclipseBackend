from setuptools import setup, Extension

module = Extension('elp2000',
                   sources=['elp2000-82bmodule.c', 'arguments.c', 'elp2000-82b.c', 'series.c'],
                   include_dirs=[],  # Adjust this if your header files are in a different directory
                   extra_compile_args=['-fPIC'])

setup(name='Elp2000Module',
      version='1.0',
      description='Python package for ELP2000-82B calculations',
      ext_modules=[module],
      package_data={'': ['elp2000.pyi']},  # Ensure the stub file is recognized as part of the package data
      include_package_data=True,
)