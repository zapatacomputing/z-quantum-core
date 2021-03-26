import setuptools
import os

setuptools.setup(
    name="z-quantum-core",
    version="0.2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="A core library of the scientific code for Orquestra.",
    url="https://github.com/zapatacomputing/z-quantum-core",
    packages=setuptools.find_namespace_packages(include=["zquantum.*"]),
    package_dir={"": "python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
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
    extras_require={
        "dev": ["pytest>=5.3.5"],
    },
)
