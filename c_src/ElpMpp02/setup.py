from setuptools import setup, Extension

module = Extension('ElpMpp02',
                   sources = ['ElpMpp02Module.cpp'],
                   include_dirs=[])

setup(
    name='ElpMpp02Module',
    version='1.0',
    description='Python package for calculating lunar coordinates using ELP2000-82B',
    ext_modules=[module]
)
