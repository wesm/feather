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

#ifndef FEATHER_METADATA_H
#define FEATHER_METADATA_H

#include <memory>
#include <string>
#include <vector>

#include "feather/compatibility.h"
#include "feather/buffer.h"
#include "feather/types.h"

namespace feather {

namespace metadata {

class TableBuilder;

class ColumnBuilder {
 public:
  ColumnBuilder(TableBuilder* parent, const std::string& name);
  ~ColumnBuilder();

  void SetValues(const ArrayMetadata& values);
  void SetCategory(const ArrayMetadata& levels, bool ordered = false);
  void SetTimestamp(TimeUnit::type unit);
  void SetTimestamp(TimeUnit::type unit, const std::string& timezone);
  void SetDate();
  void SetTime(TimeUnit::type unit);
  void SetUserMetadata(const std::string& data);

  void Finish();

 private:
  TableBuilder* parent_;
  class Impl;
  std::shared_ptr<Impl> impl_;
};

class TableBuilder {
 public:
  TableBuilder();
  explicit TableBuilder(int64_t num_rows);

  std::unique_ptr<ColumnBuilder> AddColumn(const std::string& name);
  void Finish();

  // These are accessible after calling Finish
  std::shared_ptr<Buffer> GetBuffer() const;

  void SetDescription(const std::string& description);
  void SetNumRows(int64_t num_rows);
 private:
  friend class ColumnBuilder;

  // PIMPL, to hide flatbuffers
  class Impl;
  std::shared_ptr<Impl> impl_;
};

// ----------------------------------------------------------------------
// Metadata reader interface classes

class File;
class Table;

class Column {
 public:
  Column() {}

  // Conceal flatbuffer types from the public API
  static std::shared_ptr<Column> Make(const void* fbs_column);

  std::string name() const;
  ColumnType::type type() const;

  PrimitiveType::type values_type() const;

  std::string user_metadata() const;

  const ArrayMetadata& values() const {
    return values_;
  }

 protected:
  void Init(const void* fbs_column);

  std::string name_;
  ColumnType::type type_;
  ArrayMetadata values_;

  std::string user_metadata_;
};

class CategoryColumn : public Column {
 public:
  static std::shared_ptr<Column> Make(const void* fbs_column);

  const ArrayMetadata& levels() const {
    return metadata_.levels;
  }

  bool ordered() const {
    return metadata_.ordered;
  }

 private:
  CategoryMetadata metadata_;
};

class TimestampColumn : public Column {
 public:
  static std::shared_ptr<Column> Make(const void* fbs_column);

  TimeUnit::type unit() const;
  std::string timezone() const;

 private:
  TimestampMetadata metadata_;
};

class DateColumn : public Column {
 public:
  static std::shared_ptr<Column> Make(const void* fbs_column);
};

class TimeColumn : public Column {
 public:
  static std::shared_ptr<Column> Make(const void* fbs_column);

  TimeUnit::type unit() const;

 private:
  TimeMetadata metadata_;
};

// TODO: address memory ownership issues of the buffer here
class Table {
 public:
  Table() : table_(nullptr) {}

  bool Open(const std::shared_ptr<Buffer>& buffer);

  std::string description() const;

  int version() const;

  // Optional
  bool has_description() const;

  int64_t num_rows() const;

  size_t num_columns() const;
  std::shared_ptr<Column> GetColumn(int i) const;
  std::shared_ptr<Column> GetColumnNamed(const std::string& name) const;

 private:
  std::shared_ptr<Buffer> buffer_;

  // Opaque fbs::CTable
  const void* table_;
};

} // namespace metadata

} // namespace feather

#endif // FEATHER_METADATA_H
