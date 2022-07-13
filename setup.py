"""
tablite
"""
from setuptools import setup
from pathlib import Path

# from tablite import __version__

folder = Path(__file__).parent
version = folder / "tablite" / "version.py"
exec(version.read_text())

file = "README.md"
readme = folder / file
assert isinstance(readme, Path)
assert readme.exists(), readme
with open(str(readme), encoding='utf-8') as f:
    long_description = f.read()

keywords = list({
    'tablite', 'table', 'tables', 'csv', 'txt', 'excel', 'xlsx', 'ods', 'zip', 'log',
    'any', 'all', 'filter', 'column', 'columns', 'rows', 'from', 'json', 'to',
    'inner join', 'outer join', 'left join', 'groupby', 'pivot', 'pivot table',
    'sort', 'is sorted', 'show', 'use disk', 'out-of-memory', 'list on disk',
    'stored list', 'min', 'max', 'sum', 'first', 'last', 'count', 'unique',
    'average', 'standard deviation', 'median', 'mode', 'in-memory', 'index',
    'indexing', 'product'
})

keywords.sort(key=lambda x: x.lower())

with open('requirements.txt', 'r') as fi:
    requirements = [v.rstrip('\n') for v in fi.readlines()]

setup(
    name="tablite",
    version=__version__,  
    url="https://github.com/root-11/tablite",
    license="MIT",
    author="https://github.com/root-11",
    description="multiprocessing enabled out-of-memory data analysis library for tabular data.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=keywords,
    packages=["tablite"],
    python_requires=">=3.7, <4",
    include_package_data=True,
    data_files=[(".", ["LICENSE", "README.md", "requirements.txt"])],
    platforms="any",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)


