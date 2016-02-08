## Feather file format

Here is the general structure of the file:

```
<4-byte magic number "FEA1">
<ARRAY 0>
<ARRAY 1>
...
<ARRAY n>
<METADATA>
<uint32: metadata size>
<4-byte magic number "FEA1">
```

There is a stream of arrays laid out end-to-end, with the metadata appended on
completion at the end of the file.

For the arrays themselves, the memory layout is type dependent.

1. Primitive arrays
2. Variable-length (`BINARY` and `UTF8`) arrays
3. Encoded variants of 1 and 2 (e.g. dictionary encoding)

## Primitive arrays

```
<null bitmask, optional> <values>
```

The null bitmask is byte-aligned, but contains 1 bit per value indicating null
or not null. For example, an array with length 5 containing 3 nulls followed by
2 valid values would have a single byte with the values

```
1 1 1 0 0 0 0 0
```

The values is a contiguous array with elements equal to the fixed-width byte
size.

If, in the metadata, `null_count = 0`, then the null bitmask is omitted.

## Variable-length arrays

We use the Apache Arrow encoding for storing variable-length values

```
<null bitmask, optional> <value offsets> <value data>
```

The null bitmask

## Dictionary encoding

Dictionary-encoded data is stored in the following layout.

```
<uint32: dictionary size>
<array: dictionary values>
<array: dictionary indices, int32 type>
```

The `total_bytes` stored in the metadata is the cumulative size of all of these
pieces of data.