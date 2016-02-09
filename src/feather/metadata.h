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

#include "feather/metadata_generated.h"
#include "feather/types.h"

namespace feather {

namespace metadata {

// Flatbuffers conveniences
typedef std::vector<flatbuffers::Offset<fbs::Column> > ColumnVector;
typedef std::vector<flatbuffers::Offset<fbs::CTable> > TableVector;

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

  flatbuffers::FlatBufferBuilder& fbb();

 private:
  TableBuilder* parent_;
  std::string name_;
  ArrayMetadata values_;

  std::string user_metadata_;

  // Column metadata

  // Is this a primitive type, or one of the types having metadata? Default is
  // primitive
  ColumnType::type type_;

  // Type-specific metadata union
  union {
    CategoryMetadata meta_category_;
    DateMetadata meta_date_;
    TimeMetadata meta_time_;
  };

  TimestampMetadata meta_timestamp_;

  flatbuffers::Offset<void> CreateColumnMetadata();
};

class TableBuilder {
 public:
  TableBuilder();
  TableBuilder(const std::string& name, int64_t num_rows);

  std::unique_ptr<ColumnBuilder> AddColumn(const std::string& name);
  void Finish();

  // These are accessible after calling Finish
  const void* GetBuffer() const;
  size_t BufferSize() const;

  flatbuffers::FlatBufferBuilder& fbb();

  void SetName(const std::string& name) {
    name_ = name;
  }

  void SetNumRows(int64_t num_rows) {
    num_rows_ = num_rows;
  }

 private:
  friend class ColumnBuilder;

  flatbuffers::FlatBufferBuilder fbb_;
  bool finished_;

  std::string name_;
  int64_t num_rows_;
  ColumnVector columns_;
};

// ----------------------------------------------------------------------
// Metadata reader interface classes

class File;
class Table;

class Column {
 public:
  Column() {}

  // Conceil flatbuffer types from the public API
  static std::shared_ptr<Column> Make(const void* fbs_column);

  std::string name() const;
  ColumnType::type type() const;

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

 private:
  DateMetadata metadata_;
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
  Table() : buffer_(nullptr), table_(nullptr) {}

  bool Open(const void* buffer, size_t);

  std::string name() const;
  int64_t num_rows() const;

  size_t num_columns() const;
  std::shared_ptr<Column> GetColumn(size_t i);
  std::shared_ptr<Column> GetColumnNamed(const std::string& name);

 private:
  const void* buffer_;
  const fbs::CTable* table_;
};

} // namespace metadata

} // namespace feather

#endif // FEATHER_METADATA_H
