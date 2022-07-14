# Packaging instructions for tablite for pypi

------------------------------------------
run:

```cmd
python -m build --wheel
twine check dist\*
twine upload sdist\*
```

based on [packaging guides](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)