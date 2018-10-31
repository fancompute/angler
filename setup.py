# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='fdfdpy',
    version='0.1.2',
    description='Electromagnetic Finite Difference Frequency Domain Solver',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Tyler Hughes, Momchil Minkov, Ian Williamson',
    author_email='tylerwhughes91@gmail.com',
    url='https://github.com/fancompute/fdfdpy',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
          'pyMKL',
          'numpy',
          'scipy',
          'matplotlib',
          'progressbar'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
