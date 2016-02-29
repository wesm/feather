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

#include "feather/types.h"

#include <cstring>

#include "feather/common.h"

namespace feather {

bool ArrayMetadata::Equals(const ArrayMetadata& other) const {
  return this->type == other.type &&
    this->encoding == other.encoding &&
    this->offset == other.offset &&
    this->length == other.length &&
    this->null_count == other.null_count &&
    this->total_bytes == other.total_bytes;
}

bool PrimitiveArray::Equals(const PrimitiveArray& other) const {
  // Should we even try comparing the data?
  if (this->type != other.type ||
      this->length != other.length ||
      this->null_count != other.null_count) {
    return false;
  }

  if (this->null_count > 0) {
    if (this->null_count != other.null_count ||
        memcmp(this->nulls, other.nulls, util::bytes_for_bits(this->length)) != 0) {
      return false;
    }
  }

  // TODO(wesm): variable-length dimensions
  if (IsVariableLength(this->type)) {
    // One more offset than
    if (memcmp(this->offsets, other.offsets,
            (this->length + 1) * sizeof(int32_t)) != 0) {
      return false;
    }
    size_t total_bytes = this->offsets[this->length] * ByteSize(this->type);
    if (memcmp(this->values, other.values, total_bytes) != 0) {
      return false;
    }
  } else {
    // Fixed size, get the number of bytes from the length and value size
    if (memcmp(this->values, other.values,
            this->length * ByteSize(this->type)) != 0) {
      return false;
    }
  }

  return true;
}

} // namespace feather
