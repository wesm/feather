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
#include <iostream>
#include <memory>

#include "feather/buffer.h"
#include "feather/common.h"
#include "feather/io.h"
#include "feather/status.h"

namespace feather {

TableReader::TableReader() {}

Status TableReader::Open(const std::shared_ptr<RandomAccessReader>& source) {
  source_ = source;

  int magic_size = static_cast<int>(strlen(FEATHER_MAGIC_BYTES));
  int footer_size = magic_size + sizeof(uint32_t);

  // Pathological issue where the file is smaller than
  if (source->size() < magic_size + footer_size) {
    return Status::Invalid("File is too small to be a well-formed file");
  }

  std::shared_ptr<Buffer> buffer;
  RETURN_NOT_OK(source->Read(magic_size, &buffer));

  if (memcmp(buffer->data(), FEATHER_MAGIC_BYTES, magic_size)) {
    return Status::Invalid("Not a feather file");
  }

  // Now get the footer and verify
  RETURN_NOT_OK(source->ReadAt(source->size() - footer_size, footer_size, &buffer));

  if (memcmp(buffer->data() + sizeof(uint32_t), FEATHER_MAGIC_BYTES, magic_size)) {
    return Status::Invalid("Feather file footer incomplete");
  }

  uint32_t metadata_length = *reinterpret_cast<const uint32_t*>(buffer->data());
  if (source->size() < magic_size + footer_size + metadata_length) {
    return Status::Invalid("File is smaller than indicated metadata size");
  }
  RETURN_NOT_OK(source->ReadAt(source->size() - footer_size - metadata_length,
          metadata_length, &buffer));

  if (!metadata_.Open(buffer)) {
    return Status::Invalid("Invalid file metadata");
  }

  if (metadata_.version() < kFeatherVersion) {
    std::cout << "This Feather file is old"
              << " and will not be readable beyond the 0.3.0 release"
              << std::endl;
  }

  return Status::OK();
}

Status TableReader::OpenFile(const std::string& abspath,
    std::unique_ptr<TableReader>* out) {
  auto reader = std::unique_ptr<MemoryMapReader>(new MemoryMapReader());
  RETURN_NOT_OK(reader->Open(abspath));
  std::shared_ptr<RandomAccessReader> source(reader.release());
  out->reset(new TableReader());
  return (*out)->Open(source);
}

bool TableReader::HasDescription() const {
  return metadata_.has_description();
}

std::string TableReader::GetDescription() const {
  return metadata_.description();
}

int TableReader::version() const {
  return metadata_.version();
}

int64_t TableReader::num_rows() const {
  return metadata_.num_rows();
}

int64_t TableReader::num_columns() const {
  return metadata_.num_columns();
}

// XXX: Hack for Feather 0.3.0 for backwards compatibility with old files
// Size in-file of written byte buffer
static int64_t GetOutputLength(int64_t nbytes) {
  if (kFeatherVersion < 2) {
    // Feather files < 0.3.0
    return nbytes;
  } else {
    return PaddedLength(nbytes);
  }
}

Status TableReader::GetPrimitiveArray(const ArrayMetadata& meta,
    PrimitiveArray* out) const {
  // Buffer data from the source (may or may not perform a copy depending on
  // input source)
  std::shared_ptr<Buffer> buffer;

  RETURN_NOT_OK(source_->ReadAt(meta.offset, meta.total_bytes, &buffer));

  const uint8_t* data = buffer->data();

  // If there are nulls, the null bitmask is first
  if (meta.null_count > 0) {
    out->nulls = data;
    data += GetOutputLength(util::bytes_for_bits(meta.length));
  } else {
    out->nulls = nullptr;
  }

  if (IsVariableLength(meta.type)) {
    out->offsets = reinterpret_cast<const int32_t*>(data);
    data += GetOutputLength((meta.length + 1) * sizeof(int32_t));
  }

  // TODO(wesm): dictionary encoded values

  // The value bytes are last
  out->values = data;

  out->type = meta.type;
  out->length = meta.length;
  out->null_count = meta.null_count;

  // Hold on to this data
  out->buffers.push_back(buffer);

  return Status::OK();
}

Status TableReader::GetPrimitive(std::shared_ptr<metadata::Column> col_meta,
    std::unique_ptr<Column>* out) const {
  auto values_meta = col_meta->values();
  PrimitiveArray values;
  RETURN_NOT_OK(GetPrimitiveArray(values_meta, &values));

  out->reset(new Column(col_meta->type(), col_meta, values));
  return Status::OK();
}

Status TableReader::GetCategory(std::shared_ptr<metadata::Column> col_meta,
    std::unique_ptr<Column>* out) const {
  PrimitiveArray values, levels;
  auto cat_meta = static_cast<metadata::CategoryColumn*>(col_meta.get());

  auto values_meta = cat_meta->values();
  RETURN_NOT_OK(GetPrimitiveArray(values_meta, &values));

  auto levels_meta = cat_meta->levels();
  RETURN_NOT_OK(GetPrimitiveArray(levels_meta, &levels));

  out->reset(new CategoryColumn(col_meta, values, levels,
      cat_meta->ordered()));

  return Status::OK();
}

Status TableReader::GetTimestamp(std::shared_ptr<metadata::Column> col_meta,
    std::unique_ptr<Column>* out) const {
  PrimitiveArray values;
  auto ts_meta = static_cast<metadata::TimestampColumn*>(col_meta.get());

  auto values_meta = ts_meta->values();
  RETURN_NOT_OK(GetPrimitiveArray(values_meta, &values));

  out->reset(new TimestampColumn(col_meta, values));
  return Status::OK();
}

Status TableReader::GetTime(std::shared_ptr<metadata::Column> col_meta,
    std::unique_ptr<Column>* out) const {
  PrimitiveArray values;
  auto time_meta = static_cast<metadata::TimeColumn*>(col_meta.get());

  auto values_meta = time_meta->values();
  RETURN_NOT_OK(GetPrimitiveArray(values_meta, &values));

  out->reset(new TimeColumn(col_meta, values));
  return Status::OK();
}

Status TableReader::GetColumn(int i, std::unique_ptr<Column>* out) const {
  std::shared_ptr<metadata::Column> col_meta = metadata_.GetColumn(i);
  switch (col_meta->type()) {
    case ColumnType::PRIMITIVE:
      RETURN_NOT_OK(GetPrimitive(col_meta, out));
      break;
    case ColumnType::CATEGORY:
      RETURN_NOT_OK(GetCategory(col_meta, out));
      break;
    case ColumnType::TIMESTAMP:
      RETURN_NOT_OK(GetTimestamp(col_meta, out));
      break;
    case ColumnType::DATE:
      RETURN_NOT_OK(GetPrimitive(col_meta, out));
      break;
    case ColumnType::TIME:
      RETURN_NOT_OK(GetTime(col_meta, out));
      break;
    default:
      out->reset(nullptr);
      break;
  }
  return Status::OK();
}

Status TableReader::GetColumnMetadata(int i,
    std::shared_ptr<metadata::Column>* out) const {
  *out = metadata_.GetColumn(i);
  return Status::OK();
}

} // namespace feather
