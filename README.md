[![Master Build Status](https://api.travis-ci.org/yuishihara/researchutils.svg?branch=master)](https://api.travis-ci.org/yuishihara/researchutils.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/yuishihara/researchutils/badge.svg?branch=master)](https://coveralls.io/github/yuishihara/researchutils?branch=master)
[![Documentation Status](https://readthedocs.org/projects/researchutils/badge/?version=latest)](https://researchutils.readthedocs.io/en/latest/?badge=latest)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# researchutils
Python utilities for deep learning research

# How to use
## To install the package

```bash
python setup.py install
```

or if you prefer using pip

```bash
pip install .
```

## When developing the package

It is recommended to use develop instead of install option to reflect changes in the directory

```bash
python setup.py develop
```

or if you prefer using pip

```bash
pip install -e .
```

## To run tests
```bash
python setup.py test
```

# Documents
You can find researchutils documentation [here](https://researchutils.readthedocs.io/)

# Contribution guide
New features and bug fixes are welcome. Send PRs. <br/>
This project is using GitHub flow ([See here for details](https://guides.github.com/introduction/flow/)) for development so do not try to push directly to master branch (It will be rejected anyway).

## Target python versions
Target python versions are 2.7, 3.4, 3.5 and 3.6 (as of August 2018). <br/>
Use [six](https://pythonhosted.org/six/), [future](https://pypi.org/project/future/) or any other libraries to keep compatibility among above python versions.

## Repository structure
### Module structure
Write your features under ./researchutils/ <br/>
Write your tests for the features under ./tests/ <br/>

### When writing utilities for chainer
Keep same directory structure of [original chainer](https://github.com/chainer/chainer) as much as possible under ./researchutils/chainer/. <br/>
For example, if you are writing new chainer.function,  place your new function under <br/>
./researchutils/chainer/functions/xxx/ <br/>
and write import statement in <br/>
./researchutils/chainer/functions/__init\__.py

## Write tests
When adding new feature such as function/class, always and must write test(s) unless it will be rejected.

### Where should I write my feature's tests?
When writing tests, for example for *feature_module*.py, please create test module file of name test_*feature_module_name*.py and place exactly at the same layer of your feature module. <br/>
See below.

```
├── researchutils
│   ├── __init__.py
│   ...
│   ├── your_owesome_module.py
...
└── tests
    ├── __init__.py
    ...
    ├── test_your_owesome_module.py
    │
    ...
```

## Write documents
Write documents of your new function/class/feature and explain what it does. <br/>
Writing documents are hard but helps others understanding what you implemented and enables using it.

### Style
We use numpy style docstring. When writing the docs, follow numpy style.
[See here](https://numpydoc.readthedocs.io/en/latest/) for details. 

### Language
Write your document in English.