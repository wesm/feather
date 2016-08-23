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

#include <cstring>
#include <memory>

#include "feather/common.h"
#include "feather/status.h"

namespace feather {

TableWriter::TableWriter() :
    initialized_stream_(false) {}

Status TableWriter::Open(const std::shared_ptr<OutputStream>& stream) {
  stream_ = stream;
  return Status::OK();
}

Status TableWriter::OpenFile(const std::string& abspath,
    std::unique_ptr<TableWriter>* out) {
  auto stream = std::unique_ptr<FileOutputStream>(new FileOutputStream());
  RETURN_NOT_OK(stream->Open(abspath));
  std::shared_ptr<OutputStream> sink(stream.release());
  out->reset(new TableWriter());
  return (*out)->Open(sink);
}

void TableWriter::SetDescription(const std::string& desc) {
  metadata_.SetDescription(desc);
}

void TableWriter::SetNumRows(int64_t num_rows) {
  metadata_.SetNumRows(num_rows);
}

Status TableWriter::Init() {
  if (!initialized_stream_) {
    int64_t bytes_written_unused;
    RETURN_NOT_OK(stream_->WritePadded(
            reinterpret_cast<const uint8_t*>(FEATHER_MAGIC_BYTES),
            strlen(FEATHER_MAGIC_BYTES), &bytes_written_unused));
    initialized_stream_ = true;
  }
  return Status::OK();
}

Status TableWriter::Finalize() {
  if (!initialized_stream_) {
    RETURN_NOT_OK(Init());
  }
  metadata_.Finish();

  auto buffer = metadata_.GetBuffer();

  // Writer metadata
  int64_t bytes_written;
  RETURN_NOT_OK(stream_->WritePadded(buffer->data(), buffer->size(), &bytes_written));
  uint32_t buffer_size = static_cast<uint32_t>(bytes_written);

  // Footer: metadata length, magic bytes
  RETURN_NOT_OK(stream_->Write(reinterpret_cast<const uint8_t*>(&buffer_size),
          sizeof(uint32_t)));
  RETURN_NOT_OK(stream_->Write(
          reinterpret_cast<const uint8_t*>(FEATHER_MAGIC_BYTES),
          strlen(FEATHER_MAGIC_BYTES)));

  return stream_->Close();
}

Status TableWriter::AppendPrimitive(const PrimitiveArray& values,
    ArrayMetadata* meta) {
  if (!initialized_stream_) {
    RETURN_NOT_OK(Init());
  }
  meta->type = values.type;
  meta->encoding = Encoding::PLAIN;

  RETURN_NOT_OK(stream_->Tell(&meta->offset));

  meta->length = values.length;
  meta->null_count = values.null_count;
  meta->total_bytes = 0;

  int64_t bytes_written;

  // Write the null bitmask
  if (values.null_count > 0) {
    // We assume there is one bit for each value in values.nulls, aligned on a
    // byte boundary, and we write this much data into the stream
    size_t null_bytes = util::bytes_for_bits(values.length);

    RETURN_NOT_OK(stream_->WritePadded(values.nulls, null_bytes, &bytes_written));
    meta->total_bytes += bytes_written;
  }

  size_t value_byte_size = ByteSize(values.type);
  size_t values_bytes;

  if (IsVariableLength(values.type)) {
    size_t offset_bytes = sizeof(int32_t) * (values.length + 1);

    values_bytes = values.offsets[values.length] * value_byte_size;

    // Write the variable-length offsets
    RETURN_NOT_OK(stream_->WritePadded(reinterpret_cast<const uint8_t*>(values.offsets),
            offset_bytes, &bytes_written));
    meta->total_bytes += bytes_written;
  } else {
    if (values.type == PrimitiveType::BOOL) {
      // Booleans are bit-packed
      values_bytes = util::bytes_for_bits(values.length);
    } else {
      values_bytes = values.length * value_byte_size;
    }
  }
  RETURN_NOT_OK(stream_->WritePadded(values.values, values_bytes, &bytes_written));
  meta->total_bytes += bytes_written;

  return Status::OK();
}

Status TableWriter::AppendPlain(const std::string& name,
    const PrimitiveArray& values) {
  // Prepare metadata payload
  ArrayMetadata meta;
  AppendPrimitive(values, &meta);

  // Append the metadata
  auto meta_builder = metadata_.AddColumn(name);
  meta_builder->SetValues(meta);
  meta_builder->Finish();

  return Status::OK();
}

Status TableWriter::AppendCategory(const std::string& name,
    const PrimitiveArray& values,
    const PrimitiveArray& levels, bool ordered) {

  if (!IsInteger(values.type)) {
    return Status::Invalid("Category values must be integers");
  }

  ArrayMetadata values_meta, levels_meta;

  AppendPrimitive(values, &values_meta);
  AppendPrimitive(levels, &levels_meta);

  auto meta_builder = metadata_.AddColumn(name);
  meta_builder->SetValues(values_meta);
  meta_builder->SetCategory(levels_meta, ordered);
  meta_builder->Finish();

  return Status::OK();
}

Status TableWriter::AppendTimestamp(const std::string& name,
    const PrimitiveArray& values,
    const TimestampMetadata& meta) {

  if (values.type != PrimitiveType::INT64)
    return Status::Invalid("Timestamp values must be INT64");

  ArrayMetadata values_meta;
  AppendPrimitive(values, &values_meta);

  auto meta_builder = metadata_.AddColumn(name);
  meta_builder->SetValues(values_meta);
  meta_builder->SetTimestamp(meta.unit, meta.timezone);
  meta_builder->Finish();
  return Status::OK();
}

Status TableWriter::AppendTime(const std::string& name, const PrimitiveArray& values,
    const TimeMetadata& meta) {

  if (values.type != PrimitiveType::INT64)
    return Status::Invalid("Timestamp values must be INT64");

  ArrayMetadata values_meta;
  AppendPrimitive(values, &values_meta);

  auto meta_builder = metadata_.AddColumn(name);
  meta_builder->SetValues(values_meta);
  meta_builder->SetTime(meta.unit);
  meta_builder->Finish();
  return Status::OK();
}

Status TableWriter::AppendDate(const std::string& name,
    const PrimitiveArray& values) {

  if (values.type != PrimitiveType::INT32)
    return Status::Invalid("Date values must be INT32");

  ArrayMetadata values_meta;
  AppendPrimitive(values, &values_meta);

  auto meta_builder = metadata_.AddColumn(name);
  meta_builder->SetValues(values_meta);
  meta_builder->SetDate();
  meta_builder->Finish();
  return Status::OK();
}

} // namespace feather
