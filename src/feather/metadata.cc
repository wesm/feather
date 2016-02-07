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

#include "feather/metadata.h"

#include "feather/exception.h"

namespace feather {

namespace metadata {

typedef flatbuffers::FlatBufferBuilder FBB;

// ----------------------------------------------------------------------
// Primitive array

const fbs::Type TYPE_FB_TO_FEATHER[] = {
  fbs::Type_BOOL,
  fbs::Type_INT8,
  fbs::Type_INT16,
  fbs::Type_INT32,
  fbs::Type_INT64,
  fbs::Type_UINT8,
  fbs::Type_UINT16,
  fbs::Type_UINT32,
  fbs::Type_UINT64,
  fbs::Type_FLOAT,
  fbs::Type_DOUBLE,
  fbs::Type_UTF8,
  fbs::Type_BINARY,
  fbs::Type_CATEGORY,
  fbs::Type_TIMESTAMP,
  fbs::Type_DATE,
  fbs::Type_TIME
};

static inline fbs::Type ToFlatbufferEnum(PrimitiveType::type type) {
  return TYPE_FB_TO_FEATHER[type];
}

static inline PrimitiveType::type FromFlatbufferEnum(fbs::Type type) {
  return static_cast<PrimitiveType::type>(static_cast<int>(type));
}

const fbs::Encoding ENCODING_ENUM_MAPPING[] = {
  fbs::Encoding_PLAIN,
  fbs::Encoding_DICTIONARY
};

static inline fbs::Encoding ToFlatbufferEnum(Encoding::type encoding) {
  return ENCODING_ENUM_MAPPING[encoding];
}

static inline Encoding::type FromFlatbufferEnum(fbs::Encoding enc) {
  return static_cast<Encoding::type>(static_cast<int>(enc));
}

static inline ColumnType::type ColumnTypeFromFB(fbs::TypeMetadata type) {
  switch (type) {
    case fbs::TypeMetadata_CategoryMetadata:
      return ColumnType::CATEGORY;
    case fbs::TypeMetadata_TimestampMetadata:
      return ColumnType::TIMESTAMP;
    case fbs::TypeMetadata_DateMetadata:
      return ColumnType::DATE;
    case fbs::TypeMetadata_TimeMetadata:
      return ColumnType::TIME;
    default:
      return ColumnType::PRIMITIVE;
  }
}

static inline flatbuffers::Offset<fbs::PrimitiveArray> GetPrimitiveArray(
    FBB& fbb, const PrimitiveArray& array) {
  return fbs::CreatePrimitiveArray(fbb,
      ToFlatbufferEnum(array.type),
      ToFlatbufferEnum(array.encoding),
      array.offset,
      array.length,
      array.null_count,
      array.total_bytes);
}

// ----------------------------------------------------------------------
// Category metadata

// ----------------------------------------------------------------------
// Date and time metadata

// ----------------------------------------------------------------------
// FileBuilder

FileBuilder::FileBuilder() :
    finished_(false) {}

void FileBuilder::Finish() {
  if (finished_) {
    throw FeatherException("can only call this once");
  }
  auto root = fbs::CreateFile(fbb_, fbb_.CreateVector(tables_));
  fbb_.Finish(root);
  finished_ = true;
}

const void* FileBuilder::GetBuffer() const {
  return fbb_.GetBufferPointer();
}

size_t FileBuilder::BufferSize() const {
  return fbb_.GetSize();
}

std::unique_ptr<TableBuilder> FileBuilder::AddTable(const std::string& name,
    int64_t num_rows) {
  return std::unique_ptr<TableBuilder>(new TableBuilder(this, name, num_rows));
}

// ----------------------------------------------------------------------
// TableBuilder

TableBuilder::TableBuilder(FileBuilder* parent, const std::string& name,
    int64_t num_rows) :
    parent_(parent),
    name_(name),
    num_rows_(num_rows) {}

FBB& TableBuilder::fbb() {
  return parent_->fbb_;
}

std::unique_ptr<ColumnBuilder> TableBuilder::AddColumn(const std::string& name) {
  return std::unique_ptr<ColumnBuilder>(new ColumnBuilder(this, name));
}

void TableBuilder::Finish() {
  FBB& buf = fbb();
  auto fb_table = fbs::CreateCTable(buf, buf.CreateString(name_),
      num_rows_,
      buf.CreateVector(columns_));
  parent_->tables_.push_back(fb_table);
}

// ----------------------------------------------------------------------
// ColumnBuilder

ColumnBuilder::ColumnBuilder(TableBuilder* parent, const std::string& name) :
    parent_(parent),
    name_(name),
    type_(ColumnType::PRIMITIVE) {}

FBB& ColumnBuilder::fbb() {
  return parent_->fbb();
}

void ColumnBuilder::SetValues(const PrimitiveArray& values) {
  values_ = values;
}

void ColumnBuilder::SetUserMetadata(const std::string& data) {
  user_metadata_ = data;
}

void ColumnBuilder::SetCategory(const CategoryMetadata& meta) {
  type_ = ColumnType::CATEGORY;
  meta_category_ = meta;
}

void ColumnBuilder::SetTimestamp(const TimestampMetadata& meta) {
  type_ = ColumnType::TIMESTAMP;
  meta_timestamp_ = meta;
}

void ColumnBuilder::SetDate(const DateMetadata& meta) {
  type_ = ColumnType::DATE;
  meta_date_ = meta;
}

void ColumnBuilder::SetTime(const TimeMetadata& meta) {
  type_ = ColumnType::TIME;
  meta_time_ = meta;
}

// Convert Feather enums to Flatbuffer enums

const fbs::TypeMetadata COLUMN_TYPE_ENUM_MAPPING[] = {
  fbs::TypeMetadata_NONE,              // PRIMITIVE
  fbs::TypeMetadata_CategoryMetadata,  // CATEGORY
  fbs::TypeMetadata_TimestampMetadata, // TIMESTAMP
  fbs::TypeMetadata_DateMetadata,      // DATE
  fbs::TypeMetadata_TimeMetadata       // TIME
};

fbs::TypeMetadata ToFlatbufferEnum(ColumnType::type column_type) {
  return COLUMN_TYPE_ENUM_MAPPING[column_type];
}

flatbuffers::Offset<void> ColumnBuilder::CreateColumnMetadata() {
  switch (type_) {
    case ColumnType::PRIMITIVE:
      return flatbuffers::Offset<void>();
    case ColumnType::CATEGORY:
      {
        auto cat_meta = fbs::CreateCategoryMetadata(fbb(),
            GetPrimitiveArray(fbb(), meta_category_.levels),
            meta_category_.ordered);
        return cat_meta.Union();
      }
    case ColumnType::TIMESTAMP:
    case ColumnType::DATE:
    case ColumnType::TIME:
    default:
      // null
      return flatbuffers::Offset<void>();
  }
}

void ColumnBuilder::Finish() {
  FBB& buf = fbb();

  // values
  auto values = GetPrimitiveArray(buf, values_);
  flatbuffers::Offset<void> metadata = CreateColumnMetadata();

  auto fb_column = fbs::CreateColumn(buf, buf.CreateString(name_),
      values,
      ToFlatbufferEnum(type_), // metadata_type
      metadata,
      buf.CreateString(user_metadata_));

  parent_->columns_.push_back(fb_column);
}

// ---------------------------------------------------------------------
// File

bool File::Open(const void* buffer, size_t length) {
  // Verify the buffer

  // Initiatilize the Flatbuffer interface
  file_ = fbs::GetFile(buffer);
  return true;
}

size_t File::num_tables() const {
  return file_->tables()->size();
}

std::shared_ptr<Table> File::GetTable(size_t i) {
  const fbs::CTable* fb_table = file_->tables()->Get(i);
  return std::make_shared<Table>(fb_table);
}

// ----------------------------------------------------------------------
// Table

std::string Table::name() const {
  return table_->name()->str();
}

int64_t Table::num_rows() const {
  return table_->num_rows();
}

size_t Table::num_columns() const {
  return table_->columns()->size();
}

std::shared_ptr<Column> Table::GetColumn(size_t i) {
  const fbs::Column* col = table_->columns()->Get(i);

  // Construct the right column wrapper for the logical type
  switch (col->metadata_type()) {
    case fbs::TypeMetadata_NONE:
      return std::make_shared<Column>(col);
    case fbs::TypeMetadata_CategoryMetadata:
      return std::make_shared<CategoryColumn>(col);
    case fbs::TypeMetadata_TimestampMetadata:
      return std::make_shared<TimestampColumn>(col);
    case fbs::TypeMetadata_DateMetadata:
      return std::make_shared<DateColumn>(col);
    case fbs::TypeMetadata_TimeMetadata:
      return std::make_shared<TimeColumn>(col);
    default:
      break;
  }
  // suppress compiler warning
  return std::shared_ptr<Column>(nullptr);
}

std::shared_ptr<Column> Table::GetColumnNamed(const std::string& name) {
  FeatherException::NYI("GetColumnNamed");
  return std::shared_ptr<Column>(nullptr);
}

// ----------------------------------------------------------------------
// Column

void FromFlatbuffer(const fbs::PrimitiveArray* values, PrimitiveArray* out) {
  out->type = FromFlatbufferEnum(values->type());
  out->encoding = FromFlatbufferEnum(values->encoding());
  out->offset = values->offset();
  out->length = values->length();
  out->null_count = values->null_count();
  out->total_bytes = values->total_bytes();
}

Column::Column(const fbs::Column* column) {
  column_ = column;
  FromFlatbuffer(column->values(), &values_);
}

std::string Column::name() const {
  return column_->name()->str();
}

ColumnType::type Column::type() const {
  return ColumnTypeFromFB(column_->metadata_type());
}

std::string Column::user_metadata() const {
  return column_->user_metadata()->str();
}

CategoryColumn::CategoryColumn(const fbs::Column* column) :
    Column(column) {
}

TimestampColumn::TimestampColumn(const fbs::Column* column) :
    Column(column) {
}

DateColumn::DateColumn(const fbs::Column* column) :
    Column(column) {
}

TimeColumn::TimeColumn(const fbs::Column* column) :
    Column(column) {
}

} // namespace metadata

} // namespace feather
