## Python interface to the Apache Arrow-based Feather File Format

Feather efficiently stores pandas DataFrame objects on disk. It depends on the
Apache Arrow for Python

## Installing

```shell
pip install feather-format
```

pip users note: ``feather-format`` depends on ``pyarrow`` and may not be
available on your platform via pip. If that does not work try conda-forge.

From [conda-forge][1]:

```shell
conda install feather-format -c conda-forge
```

## Limitations

Some features of pandas are not supported in Feather:

* Non-string column names
* Row indexes
* Object-type columns with non-homogeneous data

[1]: https://conda-forge.github.io