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

#include "feather/writer.h"

#include <memory>

#include "feather/common.h"

namespace feather {

TableWriter::TableWriter(std::unique_ptr<OutputStream> stream) :
    stream_(std::move(stream)) {
  stream_->Write(reinterpret_cast<const uint8_t*>(FEATHER_MAGIC_BYTES),
      strlen(FEATHER_MAGIC_BYTES));
}

void TableWriter::SetDescription(const std::string& desc) {
  metadata_.SetDescription(desc);
}

void TableWriter::SetNumRows(int64_t num_rows) {
  metadata_.SetNumRows(num_rows);
}

void TableWriter::Finalize() {
  metadata_.Finish();

  uint32_t buffer_size = metadata_.BufferSize();

  // Writer metadata
  stream_->Write(reinterpret_cast<const uint8_t*>(metadata_.GetBuffer()),
      buffer_size);

  // Footer: metadata length, magic bytes
  stream_->Write(reinterpret_cast<const uint8_t*>(&buffer_size), sizeof(uint32_t));
  stream_->Write(reinterpret_cast<const uint8_t*>(FEATHER_MAGIC_BYTES),
      strlen(FEATHER_MAGIC_BYTES));
}

void TableWriter::AppendPlain(const std::string& name,
    const PrimitiveArray& values) {
  // Prepare metadata payload
  ArrayMetadata meta;
  meta.type = values.type;
  meta.offset = stream_->Tell();
  meta.length = values.length;
  meta.null_count = values.null_count;

  // Write the null bitmask
  if (values.null_count > 0) {
    // We assume there is one bit for each value in values.nulls, aligned on a
    // byte boundary, and we write this much data into the stream
    size_t null_bytes = util::ceil_byte(values.length);

    meta.total_bytes += null_bytes;
    stream_->Write(values.nulls, null_bytes);
  }

  size_t value_byte_size = ByteSize(values.type);
  size_t values_bytes;

  if (IsVariableLength(values.type)) {
    size_t offset_bytes = sizeof(int32_t) * (values.length + 1);

    values_bytes = values.offsets[values.length] * value_byte_size + offset_bytes;

    // Write the variable-length offsets
    stream_->Write(reinterpret_cast<const uint8_t*>(values.offsets),
        offset_bytes);
  } else {
    values_bytes = values.length * value_byte_size;
  }
  stream_->Write(values.values, values_bytes);
  meta.total_bytes += values_bytes;

  // Append the metadata
  auto meta_builder = metadata_.AddColumn(name);
  meta_builder->SetValues(meta);
  meta_builder->Finish();
}

} // namespace feather
