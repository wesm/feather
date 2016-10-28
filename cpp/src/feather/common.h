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

#ifndef FEATHER_COMMON_H
#define FEATHER_COMMON_H

#include "feather/compatibility.h"

namespace feather {

static constexpr const char* FEATHER_MAGIC_BYTES = "FEA1";

static constexpr const int kFeatherDefaultAlignment = 8;
static constexpr const int kFeatherVersion = 2;

static inline int64_t PaddedLength(int64_t nbytes) {
  static const int64_t alignment = kFeatherDefaultAlignment;
  return ((nbytes + alignment - 1) / alignment) * alignment;
}

namespace util {

static inline size_t ceil_byte(size_t size) {
  return (size + 7) & ~7;
}

static inline int64_t bytes_for_bits(int64_t size) {
  return ((size + 7) & ~7) / 8;
}

static constexpr uint8_t BITMASK[] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline bool get_bit(const uint8_t* bits, int i) {
  return (bits[i / 8] & BITMASK[i % 8]) != 0;
}

static inline bool bit_not_set(const uint8_t* bits, int i) {
  return (bits[i / 8] & BITMASK[i % 8]) == 0;
}

static inline void clear_bit(uint8_t* bits, int i) {
  bits[i / 8] &= ~BITMASK[i % 8];
}

static inline void set_bit(uint8_t* bits, int i) {
  bits[i / 8] |= BITMASK[i % 8];
}

static inline void* fill_buffer(void* buffer, int value, size_t n) {
  if (buffer && n)
    ::memset(buffer, value, n);
  return buffer;
}

} // namespace util

} // namespace feather

#endif // FEATHER_COMMON_H
