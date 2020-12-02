import setuptools
import os

setuptools.setup(
    name="z-quantum-core",
    version="0.2.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="A core library of the scientific code for Orquestra.",
    url="https://github.com/zapatacomputing/z-quantum-core",
    packages=[
        "zquantum.core",
        "zquantum.core.bitstring_distribution",
        "zquantum.core.bitstring_distribution.distance_measures",
        "zquantum.core.circuit",
        "zquantum.core.history",
        "zquantum.core.interfaces",
        "zquantum.core.openfermion",
        "zquantum.core.testing",
    ],
    package_dir={"": "src/python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "pytest>=5.3.5",
        "networkx==2.4",
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "sympy>=1.5",
        "openfermion>=0.11.0",
        "openfermioncirq==0.4.0",
        "lea>=3.2.0",
        "pyquil>=2.17.0",
        "cirq==0.9.0",
        "qiskit==0.18.3",
        "quantum-grove>=1.0.0",
        "overrides>=3.1.0",
    ],
)
