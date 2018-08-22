from setuptools import find_packages
from setuptools import setup

install_requires = ['numpy', 'chainer', 'slackclient', 'matplotlib']
tests_require = ['pytest', 'mock']
setup_requires = ["pytest-runner"]

setup(
    name='researchutils',
    version='0.0.1',
    description='Python utilities for deep learning research',
    author='Yu Ishihara',
    author_email='yuishihara1225@gmail.com',
    install_requires=install_requires,
    url='https://github.com/yuishihara/researchutils',
    license='MIT License',
    packages=find_packages(exclude=('tests')),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require
)
