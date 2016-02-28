## Python interface to the Apache Arrow-based Feather File Format

## Mac notes

Anaconda uses a default 10.5 deployment target which does not have C++11
properly available. This can be fixed by setting:

```
export MACOSX_DEPLOYMENT_TARGET=10.10
```