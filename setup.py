"""
tablite
"""
build_tag = "6fbcd528ead47c1a98b8437181cbb62060709f610d2f127d7a005299540c65c8"
from setuptools import setup
from pathlib import Path


folder = Path(__file__).parent
file = "README.md"
readme = folder / file
assert isinstance(readme, Path)
assert readme.exists(), readme
with open(str(readme), encoding='utf-8') as f:
    long_description = f.read()

keywords = list({
    'table', 'tables', 'csv', 'txt', 'excel', 'xlsx', 'ods', 'zip', 'log',
    'any', 'all', 'filter', 'column', 'columns', 'rows', 'from', 'json', 'to',
    'inner join', 'outer join', 'left join', 'groupby', 'pivot', 'pivot table',
    'sort', 'is sorted', 'show', 'use disk', 'out-of-memory', 'list on disk',
    'stored list', 'min', 'max', 'sum', 'first', 'last', 'count', 'unique',
    'average', 'standard deviation', 'median', 'mode', 'in-memory', 'index'
})

keywords.sort(key=lambda x: x.lower())

with open('requirements.txt', 'r') as fi:
    requirements = [v.rstrip('\n') for v in fi.readlines()]


setup(
    name="tablite",
    version="2020.12.21.68845",
    url="https://github.com/root-11/tablite",
    license="MIT",
    author="Bjorn Madsen",
    author_email="bjorn.madsen@operationsresearchgroup.com",
    description="A table crunching library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=keywords,
    packages=["table"],
    include_package_data=True,
    data_files=[(".", ["LICENSE", "README.md"])],
    platforms="any",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)


