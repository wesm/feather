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

static inline fbs::TimeUnit ToFlatbufferEnum(TimeUnit::type unit) {
  return static_cast<fbs::TimeUnit>(static_cast<int>(unit));
}

static inline TimeUnit::type FromFlatbufferEnum(fbs::TimeUnit unit) {
  return static_cast<TimeUnit::type>(static_cast<int>(unit));
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

ColumnBuilder::~ColumnBuilder() {}

FBB& ColumnBuilder::fbb() {
  return parent_->fbb();
}

void ColumnBuilder::SetValues(const PrimitiveArray& values) {
  values_ = values;
}

void ColumnBuilder::SetUserMetadata(const std::string& data) {
  user_metadata_ = data;
}

void ColumnBuilder::SetCategory(const PrimitiveArray& levels, bool ordered) {
  type_ = ColumnType::CATEGORY;
  meta_category_.levels = levels;
  meta_category_.ordered = ordered;
}

void ColumnBuilder::SetTimestamp(TimeUnit::type unit) {
  type_ = ColumnType::TIMESTAMP;
  meta_timestamp_.unit = unit;
}

void ColumnBuilder::SetTimestamp(TimeUnit::type unit,
    const std::string& timezone) {
  SetTimestamp(unit);
  meta_timestamp_.timezone = timezone;
}

void ColumnBuilder::SetDate() {
  type_ = ColumnType::DATE;
}

void ColumnBuilder::SetTime(TimeUnit::type unit) {
  type_ = ColumnType::TIME;
  meta_time_.unit = unit;
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
      // flatbuffer void
      return 0;
    case ColumnType::CATEGORY:
      {
        auto cat_meta = fbs::CreateCategoryMetadata(fbb(),
            GetPrimitiveArray(fbb(), meta_category_.levels),
            meta_category_.ordered);
        return cat_meta.Union();
      }
    case ColumnType::TIMESTAMP:
      {
        // flatbuffer void
        flatbuffers::Offset<flatbuffers::String> tz = 0;
        if (!meta_timestamp_.timezone.empty()) {
          tz = fbb().CreateString(meta_timestamp_.timezone);
        }

        auto ts_meta = fbs::CreateTimestampMetadata(fbb(),
            ToFlatbufferEnum(meta_timestamp_.unit), tz);
        return ts_meta.Union();
      }
    case ColumnType::DATE:
      {
        auto date_meta = fbs::CreateDateMetadata(fbb());
        return date_meta.Union();
      }
    case ColumnType::TIME:
      {
        auto time_meta = fbs::CreateTimeMetadata(fbb(),
            ToFlatbufferEnum(meta_time_.unit));
        return time_meta.Union();
      }
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
      return Column::Make(col);
    case fbs::TypeMetadata_CategoryMetadata:
      return CategoryColumn::Make(col);
    case fbs::TypeMetadata_TimestampMetadata:
      return TimestampColumn::Make(col);
    case fbs::TypeMetadata_DateMetadata:
      return DateColumn::Make(col);
    case fbs::TypeMetadata_TimeMetadata:
      return TimeColumn::Make(col);
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

void FromFlatbuffer(const fbs::PrimitiveArray* values, PrimitiveArray& out) {
  out.type = FromFlatbufferEnum(values->type());
  out.encoding = FromFlatbufferEnum(values->encoding());
  out.offset = values->offset();
  out.length = values->length();
  out.null_count = values->null_count();
  out.total_bytes = values->total_bytes();
}

void Column::Init(const void* fbs_column) {
  const fbs::Column* column = static_cast<const fbs::Column*>(fbs_column);
  name_ = column->name()->str();
  type_ = ColumnTypeFromFB(column->metadata_type());
  FromFlatbuffer(column->values(), values_);

  auto user_meta = column->user_metadata();
  if (user_meta->size() > 0) {
    user_metadata_ = user_meta->str();
  }
}

std::shared_ptr<Column> Column::Make(const void* fbs_column) {
  auto result = std::make_shared<Column>();
  result->Init(fbs_column);
  return result;
}

std::string Column::name() const {
  return name_;
}

ColumnType::type Column::type() const {
  return type_;
}

std::string Column::user_metadata() const {
  return user_metadata_;
}

// ----------------------------------------------------------------------
// Category column

std::shared_ptr<Column> CategoryColumn::Make(const void* fbs_column) {
  const fbs::Column* column = static_cast<const fbs::Column*>(fbs_column);

  auto result = std::make_shared<CategoryColumn>();
  result->Init(fbs_column);

  // Category metadata
  auto meta = static_cast<const fbs::CategoryMetadata*>(column->metadata());
  FromFlatbuffer(meta->levels(), result->metadata_.levels);
  result->metadata_.ordered = meta->ordered();
  return result;
}

// ----------------------------------------------------------------------
// Timestamp column

std::shared_ptr<Column> TimestampColumn::Make(const void* fbs_column) {
  const fbs::Column* column = static_cast<const fbs::Column*>(fbs_column);

  auto result = std::make_shared<TimestampColumn>();
  result->Init(fbs_column);

  auto meta = static_cast<const fbs::TimestampMetadata*>(column->metadata());
  result->metadata_.unit = FromFlatbufferEnum(meta->unit());

  auto tz = meta->timezone();
  // flatbuffer non-null
  if (tz != 0) {
    result->metadata_.timezone = tz->str();
  }

  return result;
}

TimeUnit::type TimestampColumn::unit() const {
  return metadata_.unit;
}

std::string TimestampColumn::timezone() const {
  return metadata_.timezone;
}

// ----------------------------------------------------------------------
// Date column

std::shared_ptr<Column> DateColumn::Make(const void* fbs_column) {
  auto result = std::make_shared<DateColumn>();
  result->Init(fbs_column);
  return result;
}

std::shared_ptr<Column> TimeColumn::Make(const void* fbs_column) {
  const fbs::Column* column = static_cast<const fbs::Column*>(fbs_column);

  auto result = std::make_shared<TimeColumn>();
  result->Init(fbs_column);

  auto meta = static_cast<const fbs::TimeMetadata*>(column->metadata());
  result->metadata_.unit = FromFlatbufferEnum(meta->unit());

  return result;
}

TimeUnit::type TimeColumn::unit() const {
  return metadata_.unit;
}

} // namespace metadata

} // namespace feather
