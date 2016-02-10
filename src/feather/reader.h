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

#include "feather/io.h"
#include "feather/metadata.h"
#include "feather/types.h"

namespace feather {

struct Column {
  Column(const std::string& name, ColumnType::type type) :
      name(name), type(type) {}

  std::string name;
  ColumnType::type type;
};

struct PrimitiveColumn : public Column {
  explicit PrimitiveColumn(const std::string& name) :
      Column(name, ColumnType::PRIMITIVE) {}

  std::shared_ptr<metadata::Column> metadata;
  PrimitiveArray values;
};

struct CategoryColumn : public Column {
  explicit CategoryColumn(const std::string& name) :
      Column(name, ColumnType::CATEGORY) {}

  std::shared_ptr<metadata::CategoryColumn> metadata;
  CategoryArray values;
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

  std::unique_ptr<Column> GetColumn(size_t i);

 private:
  std::shared_ptr<RandomAccessReader> source_;
  metadata::Table metadata_;
};

} // namespace feather

#endif // FEATHER_WRITER_H
