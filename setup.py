# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

dependencies = [
        'pyMKL',
        'numpy',
        'scipy',
        'matplotlib>=2.2.2',
        'progressbar2==3.37.1',
        'autograd',
        'future'
]

setup(
    name='angler',
    version='0.0.15',
    description='Adjoint Nonlinear Gradients',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Tyler Hughes, Momchil Minkov, Ian Williamson',
    author_email='tylerwhughes91@gmail.com',
    url='https://github.com/fancompute/angler',
    packages=find_packages(),
    install_requires=dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
