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
        "pytest>=5.3.5",
        "networkx==2.4",
        "numpy>=1.18.1",
        "sympy>=1.5",
        "scipy>=1.4.1",
        "openfermion>=1.0.0",
        "openfermioncirq>=0.4.0",
        "lea>=3.2.0",
        "pyquil>=2.17.0",
        "cirq>=0.9.0",
        "qiskit==0.23.2",
        "quantum-grove>=1.0.0",
        "overrides>=3.1.0",
    ],
)
