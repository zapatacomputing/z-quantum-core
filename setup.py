import os

import setuptools

dev_requires = [
    "pytest>=3.7.1",
    "pytest-cov>=2.5.1",
    "tox>=3.2.1",
    "flake8>=3.7.9",
    "black>=19.3b0",
    "pre_commit>=2.10.1",
    "mypy==0.812",
    "isort>=5.8",
]

extras_require = {
    "develop": dev_requires,
}

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
    install_requires=[
        "networkx==2.4",
        "numpy>=1.20",
        "scipy>=1.4.1",
        "sympy>=1.5",
        "openfermion~=1.0",
        "lea>=3.2.0",
        "overrides~=3.1",
        "python-rapidjson",
    ],
    extras_require=extras_require,
    setup_requires=["setuptools_scm~=6.0"],
    # Without this, users of this library would get mypy errors. See also:
    # https://github.com/python/mypy/issues/7508#issuecomment-531965557
    zip_safe=False,
)
