"""
tablite
"""
from setuptools import setup
from pathlib import Path
import nimporter

__version__ = None
version_file = Path(__file__).parent / "tablite" / "version.py"
exec(version_file.read_text())
assert isinstance(__version__, str)

readme = Path(__file__).parent / "README.md"
assert isinstance(readme, Path)
assert readme.exists(), readme
with open(str(readme), encoding="utf-8") as f:
    long_description = f.read()

keywords = list(
    {
        "tablite",
        "table",
        "tables",
        "csv",
        "txt",
        "excel",
        "xlsx",
        "ods",
        "zip",
        "log",
        "any",
        "all",
        "filter",
        "column",
        "columns",
        "rows",
        "from",
        "json",
        "to",
        "inner join",
        "outer join",
        "left join",
        "groupby",
        "pivot",
        "pivot table",
        "sort",
        "is sorted",
        "show",
        "use disk",
        "out-of-memory",
        "list on disk",
        "stored list",
        "min",
        "max",
        "sum",
        "first",
        "last",
        "count",
        "unique",
        "average",
        "standard deviation",
        "median",
        "mode",
        "in-memory",
        "index",
        "indexing",
        "product",
        "replace missing values",
        "data imputation",
        "imputation",
        "date range",
        "read csv",
        "xround",
        "guess",
        "remove duplicates",
        "replace",
        "to_pandas",
        "pandas",
        "from_pandas",
        "transpose",
        "dict",
        "list",
        "numpy",
        "tools",
    }
)

keywords.sort(key=lambda x: x.lower())

with open("requirements.txt", "r") as fi:
    requirements = [v.rstrip("\n") for v in fi.readlines()]

setup(
    name="tablite",
    version=__version__,
    url="https://github.com/root-11/tablite",
    license="MIT",
    author="https://github.com/root-11",
    description="multiprocessing enabled out-of-memory data analysis library for tabular data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=keywords,
    packages=["tablite", "_nimlite"],
     package_data={
        "_nimlite": ["**/*.nim"],
    },
    python_requires=">=3.8",
    include_package_data=True,
    data_files=[(".", ["LICENSE", "README.md", "requirements.txt"])],
    platforms="any",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
