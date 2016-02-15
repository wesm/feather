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

#ifndef FEATHER_READER_H
#define FEATHER_READER_H

#include <string>

#include "feather/buffer.h"
#include "feather/io.h"
#include "feather/metadata.h"
#include "feather/types.h"

namespace feather {

class Column {
 public:
  Column(ColumnType::type type, std::shared_ptr<metadata::Column> metadata,
      const PrimitiveArray& values, std::shared_ptr<Buffer> buffer) :
      type_(type),
      metadata_(metadata),
      buffer_(buffer),
      values_(values) {}

  const PrimitiveArray& values() const {
    return values_;
  }

  const std::shared_ptr<metadata::Column>& metadata() const {
    return metadata_;
  }

 protected:
  ColumnType::type type_;
  std::shared_ptr<metadata::Column> metadata_;
  std::shared_ptr<Buffer> buffer_;
  PrimitiveArray values_;
};

class CategoryColumn : public Column {
 public:
  CategoryColumn(std::shared_ptr<metadata::Column> metadata,
      const PrimitiveArray& values,
      std::shared_ptr<Buffer> values_buffer,
      const PrimitiveArray& levels,
      std::shared_ptr<Buffer> levels_buffer) :
      Column(ColumnType::CATEGORY, metadata, values, values_buffer),
      levels_(levels),
      levels_buffer_(levels_buffer) {
    category_meta_ = static_cast<const metadata::CategoryColumn*>(metadata.get());
  }

  const PrimitiveArray& levels() const {
    return levels_;
  }

 private:
  const metadata::CategoryColumn* category_meta_;
  PrimitiveArray levels_;
  std::shared_ptr<Buffer> levels_buffer_;
};

class TableReader {
 public:
  explicit TableReader(std::shared_ptr<RandomAccessReader> source);

  static std::unique_ptr<TableReader> OpenFile(const std::string& abspath);

  // Optional table description
  //
  // This does not return a const std::string& because a string has to be
  // copied from the flatbuffer to be able to return a non-flatbuffer type
  std::string GetDescription() const;
  bool HasDescription() const;

  int64_t num_rows() const;
  int64_t num_columns() const;

  std::shared_ptr<Column> GetColumn(int i);

 private:
  // Retrieve a primitive array from the data source
  //
  // @returns: a Buffer instance, the precise type will depend on the kind of
  // input data source (which may or may not have memory-map like semantics)
  std::shared_ptr<Buffer> GetPrimitiveArray(const ArrayMetadata& meta,
      PrimitiveArray* out);

  std::shared_ptr<RandomAccessReader> source_;
  metadata::Table metadata_;
};

} // namespace feather

#endif // FEATHER_WRITER_H
