# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()

dependencies = [
        'MKL',
        'pyMKL',
        'numpy',
        'scipy',
        'matplotlib>=2.2.2',
        'progressbar2',
        'autograd',
        'future',
        'billiard',
        'celery',
        'kombu',
        'amqp'
]

setup(
    name='angler',
    version='0.0.9',
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
)
