## Feather: fast, interoperable data frame storage for Python and R

## Development

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
smart pointers and manage the lifetime of objects and memory, and generally use
[RAII][2] whenever possible.

## License and Copyrights

This library is released under the Apache License, Version 2.0.

See `NOTICE` for details about the library's copyright holders.

[1]: http://google.github.io/styleguide/cppguide.html
[2]: https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization