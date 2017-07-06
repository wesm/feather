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
(0) or not null (1) (this is also how [PostgreSQL stores nulls][2]). We use LSB
right-to-left bit-numbering ([reference][1]). For example, an array with length
6 with [valid, valid, null, valid, null, valid] would have a single byte with
the values

```
0 0 1 0 1 0 1 1
```

In C, the code to check a bit looks like:

```
bits[i / 8] & (1 << (i % 8))
```

The values is a contiguous array with elements equal to the fixed-width byte
size.

If, in the metadata, `null_count = 0`, then the null bitmask is omitted.

## Variable-length arrays

We use the [Apache Arrow][3] encoding for storing variable-length values

```
<null bitmask, optional> <int32_t* value offsets> <uint8_t* value data>
```

For an array of `N` elements, `N+1` offsets are written so the end of last value can be determined.

## Dictionary encoding

Dictionary-encoded data is stored in the following layout.

```
<uint32: dictionary size>
<array: dictionary values>
<array: dictionary indices, int32 type>
```

The `total_bytes` stored in the metadata is the cumulative size of all of these
pieces of data.

[1]: https://en.wikipedia.org/wiki/Bit_numbering
[2]: http://www.postgresql.org/docs/9.5/static/storage-page-layout.html
[3]: http://github.com/apache/arrow
