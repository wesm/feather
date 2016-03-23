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

class Status;

class Column {
 public:
  Column(ColumnType::type type,
      const std::shared_ptr<metadata::Column>& metadata,
      const PrimitiveArray& values) :
      type_(type),
      metadata_(metadata),
      values_(values) {}

  const PrimitiveArray& values() const {
    return values_;
  }

  ColumnType::type type() const {
    return type_;
  }

  const std::shared_ptr<metadata::Column>& metadata() const {
    return metadata_;
  }

  std::string name() const {
    return metadata_->name();
  }

 protected:
  ColumnType::type type_;
  std::shared_ptr<metadata::Column> metadata_;
  PrimitiveArray values_;
};

class CategoryColumn : public Column {
 public:
  CategoryColumn(const std::shared_ptr<metadata::Column>& metadata,
      const PrimitiveArray& values,
      const PrimitiveArray& levels) :
      Column(ColumnType::CATEGORY, metadata, values),
      levels_(levels) {
    category_meta_ = static_cast<const metadata::CategoryColumn*>(metadata.get());
  }

  const PrimitiveArray& levels() const {
    return levels_;
  }

 private:
  const metadata::CategoryColumn* category_meta_;
  PrimitiveArray levels_;
};

class TimestampColumn : public Column {
 public:
  TimestampColumn(const std::shared_ptr<metadata::Column>& metadata,
      const PrimitiveArray& values) :
      Column(ColumnType::TIMESTAMP, metadata, values)  {
    timestamp_meta_ = static_cast<const metadata::TimestampColumn*>(metadata.get());
  }

  TimeUnit::type unit() const {
    return timestamp_meta_->unit();
  }

  std::string timezone() const {
    return timestamp_meta_->timezone();
  }

 private:
  const metadata::TimestampColumn* timestamp_meta_;
};

class DateColumn : public Column {
 public:
  DateColumn(const std::shared_ptr<metadata::Column>& metadata,
      const PrimitiveArray& values) :
      Column(ColumnType::DATE, metadata, values)  {
    date_meta_ = static_cast<const metadata::DateColumn*>(metadata.get());
  }

 private:
  const metadata::DateColumn* date_meta_;
};

class TimeColumn : public Column {
 public:
  TimeColumn(const std::shared_ptr<metadata::Column>& metadata,
      const PrimitiveArray& values) :
      Column(ColumnType::TIME, metadata, values)  {
    time_meta_ = static_cast<const metadata::TimeColumn*>(metadata.get());
  }

  TimeUnit::type unit() const {
    return time_meta_->unit();
  }

 private:
  const metadata::TimeColumn* time_meta_;
};

class TableReader {
 public:
  TableReader();

  Status Open(const std::shared_ptr<RandomAccessReader>& source);

  static Status OpenFile(const std::string& abspath, std::unique_ptr<TableReader>* out);

  // Optional table description
  //
  // This does not return a const std::string& because a string has to be
  // copied from the flatbuffer to be able to return a non-flatbuffer type
  std::string GetDescription() const;
  bool HasDescription() const;

  int64_t num_rows() const;
  int64_t num_columns() const;

  Status GetColumn(int i, std::shared_ptr<Column>* out);

 private:
  Status GetPrimitive(std::shared_ptr<metadata::Column> col_meta,
      std::shared_ptr<Column>* out);
  Status GetCategory(std::shared_ptr<metadata::Column> col_meta,
      std::shared_ptr<Column>* out);

  Status GetTimestamp(std::shared_ptr<metadata::Column> col_meta,
      std::shared_ptr<Column>* out);

  Status GetTime(std::shared_ptr<metadata::Column> col_meta,
      std::shared_ptr<Column>* out);

  Status GetDate(std::shared_ptr<metadata::Column> col_meta,
      std::shared_ptr<Column>* out);

  // Retrieve a primitive array from the data source
  //
  // @returns: a Buffer instance, the precise type will depend on the kind of
  // input data source (which may or may not have memory-map like semantics)
  Status GetPrimitiveArray(const ArrayMetadata& meta, PrimitiveArray* out);

  std::shared_ptr<RandomAccessReader> source_;
  metadata::Table metadata_;
};

} // namespace feather

#endif // FEATHER_WRITER_H
