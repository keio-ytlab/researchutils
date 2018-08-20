from setuptools import find_packages
from setuptools import setup

install_requires = ['']
test_requires = ['']

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
    test_suite='tests',
    test_requires=test_requires
)