import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'karman'
MINIMUM_PYTHON_VERSION = 3, 9

with open('README.md', 'r') as f:
    long_description = f.read()

def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
setup(
    name='karman',
    version=read_package_variable('__version__'),
    description='Karman: Data Driven Thermospheric Density Modeling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='',
    author_email='giacomo.acciarini@gmail.com',
    packages=find_packages(),
    url='https://github.com/spaceml-org/karman',
    install_requires=['numpy', 'torch','matplotlib','scikit-learn','pandas','tables','tqdm','pyfiglet>=0.8.0','termcolor','wandb','pyatmos','spaceweather','nrlmsise00','tft-torch'],
    extras_require={'dev': ['pytest', 'coverage', 'pytest-xdist','netcdf4','cftime','flake8']},
    classifiers=['License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 'Programming Language :: Python :: 3'],
    include_package_data=True,
    package_data={
        "karman": ["*.pk", "*.csv"],  # Include all CSV files in the karman directory
    },
)
