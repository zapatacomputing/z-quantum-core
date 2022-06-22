################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
import os
import site
import sys
import warnings
from platform import python_version

import setuptools

try:
    from subtrees.z_quantum_actions.setup_extras import extras
except ImportError:
    warnings.warn("Unable to import extras")
    extras = {}

# Workaound for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]


def _this_path():
    return os.path.abspath(os.path.dirname(__file__))


def _read_readme():
    with open(os.path.join(_this_path(), "README.md")) as f:
        return f.read()


setuptools.setup(
    name="z-quantum-core",
    use_scm_version=True,
    license="Apache-2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="A core library of the scientific code for Orquestra.",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/z-quantum-core",
    packages=setuptools.find_namespace_packages(
        include=["zquantum.*"], where="src/python"
    ),
    package_dir={"": "src/python"},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    # Avoid bug in Protocol in python 3.9.7. Note that higher and lower versions of Python are fine.
    python_version="!=3.9.7",
    install_requires=[
        "networkx==2.4",
        "numpy>=1.20",
        "scipy>=1.4.1",
        "sympy>=1.5,<=1.9",
        "overrides~=3.1",
        "python-rapidjson",
        "h5py>=2.8",
    ],
    extras_require=extras,
    # Without this, users of this library would get mypy errors. See also:
    # https://github.com/python/mypy/issues/7508#issuecomment-531965557
    zip_safe=False,
)
