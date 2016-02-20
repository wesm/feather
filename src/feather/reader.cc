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

#include "feather/reader.h"

#include <cstring>
#include <memory>

#include "feather/buffer.h"
#include "feather/common.h"
#include "feather/exception.h"
#include "feather/io.h"

namespace feather {

TableReader::TableReader(const std::shared_ptr<RandomAccessReader>& source) :
    source_(source) {
  int magic_size = strlen(FEATHER_MAGIC_BYTES);
  int footer_size = magic_size + sizeof(uint32_t);

  // Pathological issue where the file is smaller than
  if (source->size() < magic_size + footer_size) {
    throw FeatherException("File is too small to be a well-formed file");
  }

  std::shared_ptr<Buffer> buffer = source->Read(magic_size);

  if (memcmp(buffer->data(), FEATHER_MAGIC_BYTES, magic_size)) {
    throw FeatherException("Not a feather file");
  }

  // Now get the footer and verify
  buffer = source->ReadAt(source->size() - footer_size, footer_size);

  if (memcmp(buffer->data() + sizeof(uint32_t), FEATHER_MAGIC_BYTES, magic_size)) {
    throw FeatherException("Feather file footer incomplete");
  }

  uint32_t metadata_length = *reinterpret_cast<const uint32_t*>(buffer->data());
  if (source->size() < magic_size + footer_size + metadata_length) {
    throw FeatherException("File is smaller than indicated metadata size");
  }
  buffer = source->ReadAt(source->size() - footer_size - metadata_length,
      metadata_length);

  if (!metadata_.Open(buffer)) {
    throw FeatherException("Invalid file metadata");
  }
}

bool TableReader::HasDescription() const {
  return metadata_.has_description();
}

std::string TableReader::GetDescription() const {
  return metadata_.description();
}

int64_t TableReader::num_rows() const {
  return metadata_.num_rows();
}

int64_t TableReader::num_columns() const {
  return metadata_.num_columns();
}

std::shared_ptr<Buffer> TableReader::GetPrimitiveArray(
    const ArrayMetadata& meta, PrimitiveArray* out) {

  // Buffer data from the source (may or may not perform a copy depending on
  // input source)
  std::shared_ptr<Buffer> buffer = source_->ReadAt(meta.offset,
      meta.total_bytes);

  const uint8_t* data = buffer->data();

  // If there are nulls, the null bitmask is first
  if (meta.null_count > 0) {
    out->nulls = data;
    data += util::ceil_byte(meta.length);
  } else {
    out->nulls = nullptr;
  }

  if (IsVariableLength(meta.type)) {
    out->offsets = reinterpret_cast<const int32_t*>(data);
    data += (meta.length + 1) * sizeof(int32_t);
  }

  // TODO(wesm): dictionary encoded values

  // The value bytes are last
  out->values = data;

  out->type = meta.type;
  out->length = meta.length;
  out->null_count = meta.null_count;

  return buffer;
}

std::shared_ptr<Column> TableReader::GetColumn(int i) {
  auto col_meta = metadata_.GetColumn(i);
  auto values_meta = col_meta->values();

  PrimitiveArray values;
  std::shared_ptr<Buffer> buffer = GetPrimitiveArray(values_meta, &values);

  switch (col_meta->type()) {
    case ColumnType::PRIMITIVE:
      return std::make_shared<Column>(ColumnType::PRIMITIVE, col_meta, values,
          buffer);
    default:
      return std::shared_ptr<Column>(nullptr);
      break;
  }
}

} // namespace feather
