import os

import setuptools


def _read_readme():
    this_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_path, "README.md"), encoding="utf-8") as f:
        return f.read()


setuptools.setup(
    name="z-quantum-core",
    version="0.2.0.dev4",
    license="Apache-2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="A core library of the scientific code for Orquestra.",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/z-quantum-core",
    packages=setuptools.find_namespace_packages(include=["zquantum.*"]),
    package_dir={"": "python"},
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "pytest>=5.3.5",
        "networkx==2.4",
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "sympy>=1.7",
        "openfermion>=1.0.0",
        "openfermioncirq==0.4.0",
        "lea>=3.2.0",
        "pyquil~=2.25",
        "cirq>=0.9.1",
        "qiskit~=0.24",
        "overrides>=3.1.0",
    ],
)
