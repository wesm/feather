// Copyright 2016 Feather Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FEATHER_TYPES_H
#define FEATHER_TYPES_H

#include <string>

namespace feather {

// Feather enums, decoupled from some of the unpleasantness of
// flatbuffers. This is also here so we can choose later to hide the
// flatbuffers dependency to C++ users of libfeather (otherwise they have to
// include/link libflatbuffers.a)
struct PrimitiveType {
  enum type {
    BOOL = 0,

    INT8 = 1,
    INT16 = 2,
    INT32 = 3,
    INT64 = 4,

    UINT8 = 5,
    UINT16 = 6,
    UINT32 = 7,
    UINT64 = 8,

    FLOAT = 9,
    DOUBLE = 10,

    UTF8 = 11,

    BINARY = 12,

    CATEGORY = 13,

    TIMESTAMP = 14,
    DATE = 15,
    TIME = 16
  };
};

struct ColumnType {
  enum type {
    PRIMITIVE,
    CATEGORY,
    TIMESTAMP,
    DATE,
    TIME
  };
};

struct Encoding {
  enum type {
    PLAIN = 0,
    /// Data is stored dictionary-encoded
    /// dictionary size: <INT32 Dictionary size>
    /// dictionary data: <TYPE primitive array>
    /// dictionary index: <INT32 primitive array>
    ///
    /// TODO: do we care about storing the index values in a smaller typeclass
    DICTIONARY = 1
  };
};

struct TimeUnit {
  enum type {
    SECOND = 0,
    MILLISECOND = 1,
    MICROSECOND = 2,
    NANOSECOND = 3
  };
};

struct PrimitiveArray {
  PrimitiveArray() {}

  PrimitiveArray(PrimitiveType::type type, Encoding::type encoding,
      int64_t offset, int64_t length, int64_t null_count,
      int64_t total_bytes) :
      type(type), encoding(encoding),
      offset(offset), length(length),
      null_count(null_count), total_bytes(total_bytes) {}

  PrimitiveType::type type;
  Encoding::type encoding;
  int64_t offset;
  int64_t length;
  int64_t null_count;
  int64_t total_bytes;
};

struct CategoryMetadata {
  PrimitiveArray levels;
  bool ordered;
};

struct TimestampMetadata {
  TimeUnit::type unit;

  // A timezone name known to the Olson timezone database. For display purposes
  // because the actual data is all UTC
  std::string timezone;
};

struct DateMetadata {
};

struct TimeMetadata {
  TimeUnit::type unit;
};

} // namespace feather

#endif // FEATHER_TYPES_H
