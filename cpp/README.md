## Feather core C++ library

## Development info

The core libfeather library is implemented in C++11.

We use [Google C++ coding style][1] with a few changes:

- We use C++ exceptions for error handling
- Long lines are permitted up to 90 characters
- We do not encourage anonymous namespaces

Style is checked with `cpplint`, after generating the make files you can
veryify the style with

```
make lint
```

Explicit memory management and non-trivial destructors are to be avoided. Use
smart pointers to manage the lifetime of objects and memory, and generally use
[RAII][2] whenever possible.
