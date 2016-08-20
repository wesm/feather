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

#include <cstdint>

#include "feather/buffer.h"
#include "feather/common.h"
#include "feather/metadata_generated.h"
#include "feather/status.h"

namespace feather {

namespace metadata {

typedef flatbuffers::FlatBufferBuilder FBB;

typedef flatbuffers::Offset<flatbuffers::String> FBString;
typedef std::vector<flatbuffers::Offset<fbs::Column>> ColumnVector;

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
    FBB& fbb, const ArrayMetadata& array) {
  return fbs::CreatePrimitiveArray(fbb,
      ToFlatbufferEnum(array.type),
      ToFlatbufferEnum(array.encoding),
      array.offset,
      array.length,
      array.null_count,
      array.total_bytes);
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

// ----------------------------------------------------------------------
// TableBuilder

class TableBuilder::Impl {
 public:
  explicit Impl(int64_t num_rows) :
      finished_(false),
      num_rows_(num_rows) {}

  FBB& fbb() {
    return fbb_;
  }

  Status Finish() {
    if (finished_) {
      return Status::Invalid("can only call this once");
    }

    FBString desc = 0;
    if (!description_.empty()) {
      desc = fbb_.CreateString(description_);
    }

    flatbuffers::Offset<flatbuffers::String> metadata = 0;

    auto root = fbs::CreateCTable(fbb_,
        desc,
        num_rows_,
        fbb_.CreateVector(columns_),
        kFeatherVersion, metadata);
    fbb_.Finish(root);
    finished_ = true;

    return Status::OK();
  }

  void set_description(const std::string& description) {
    description_ = description;
  }

  void set_num_rows(int64_t num_rows) {
    num_rows_ = num_rows;
  }

  void add_column(const flatbuffers::Offset<fbs::Column>& col) {
    columns_.push_back(col);
  }

 private:
  flatbuffers::FlatBufferBuilder fbb_;
  ColumnVector columns_;

  bool finished_;
  std::string description_;
  int64_t num_rows_;
};

TableBuilder::TableBuilder(int64_t num_rows) {
  impl_.reset(new Impl(num_rows));
}

TableBuilder::TableBuilder() {
  impl_.reset(new Impl(0));
}

std::shared_ptr<Buffer> TableBuilder::GetBuffer() const {
  return std::make_shared<Buffer>(impl_->fbb().GetBufferPointer(),
      static_cast<int64_t>(impl_->fbb().GetSize()));
}

std::unique_ptr<ColumnBuilder> TableBuilder::AddColumn(const std::string& name) {
  return std::unique_ptr<ColumnBuilder>(new ColumnBuilder(this, name));
}

void TableBuilder::SetDescription(const std::string& description) {
  impl_->set_description(description);
}

void TableBuilder::SetNumRows(int64_t num_rows) {
  impl_->set_num_rows(num_rows);
}

void TableBuilder::Finish() {
  impl_->Finish();
}

// ----------------------------------------------------------------------
// ColumnBuilder

class ColumnBuilder::Impl {
 public:
  Impl(FBB* builder, const std::string& name) :
      name_(name),
      type_(ColumnType::PRIMITIVE) {
    fbb_ = builder;
  }

  flatbuffers::Offset<void> CreateColumnMetadata() {
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

  flatbuffers::Offset<fbs::Column> Finish() {
    FBB& buf = fbb();

    // values
    auto values = GetPrimitiveArray(buf, values_);
    flatbuffers::Offset<void> metadata = CreateColumnMetadata();

    return fbs::CreateColumn(buf, buf.CreateString(name_),
        values,
        ToFlatbufferEnum(type_), // metadata_type
        metadata,
        buf.CreateString(user_metadata_));
  }

  void set_values(const ArrayMetadata& values) {
    values_ = values;
  }

  void set_user_metadata(const std::string& data) {
    user_metadata_ = data;
  }

  void set_category(const ArrayMetadata& levels, bool ordered) {
    type_ = ColumnType::CATEGORY;
    meta_category_.levels = levels;
    meta_category_.ordered = ordered;
  }

  void set_timestamp(TimeUnit::type unit) {
    type_ = ColumnType::TIMESTAMP;
    meta_timestamp_.unit = unit;
  }

  void set_timestamp(TimeUnit::type unit,
      const std::string& timezone) {
    set_timestamp(unit);
    meta_timestamp_.timezone = timezone;
  }

  void set_date() {
    type_ = ColumnType::DATE;
  }

  void set_time(TimeUnit::type unit) {
    type_ = ColumnType::TIME;
    meta_time_.unit = unit;
  }

  FBB& fbb() {
    return *fbb_;
  }

 private:
  std::string name_;
  ArrayMetadata values_;
  std::string user_metadata_;

  // Column metadata

  // Is this a primitive type, or one of the types having metadata? Default is
  // primitive
  ColumnType::type type_;

  // Type-specific metadata union
  CategoryMetadata meta_category_;
  // DateMetadata meta_date_; // not used?
  TimeMetadata meta_time_;

  TimestampMetadata meta_timestamp_;

  FBB* fbb_;
};

void ColumnBuilder::Finish() {
  auto result = impl_->Finish();
  // horrible coupling, but can clean this up later
  parent_->impl_->add_column(result);
}

ColumnBuilder::ColumnBuilder(TableBuilder* parent, const std::string& name) :
    parent_(parent) {
  impl_.reset(new Impl(&parent->impl_->fbb(), name));
}

ColumnBuilder::~ColumnBuilder() {}

void ColumnBuilder::SetValues(const ArrayMetadata& values) {
  impl_->set_values(values);
}

void ColumnBuilder::SetUserMetadata(const std::string& data) {
  impl_->set_user_metadata(data);
}

void ColumnBuilder::SetCategory(const ArrayMetadata& levels, bool ordered) {
  impl_->set_category(levels, ordered);
}

void ColumnBuilder::SetTimestamp(TimeUnit::type unit) {
  impl_->set_timestamp(unit);
}

void ColumnBuilder::SetTimestamp(TimeUnit::type unit,
    const std::string& timezone) {
  impl_->set_timestamp(unit, timezone);
}

void ColumnBuilder::SetDate() {
  impl_->set_date();
}

void ColumnBuilder::SetTime(TimeUnit::type unit) {
  impl_->set_time(unit);
}

// ----------------------------------------------------------------------
// Table

bool Table::Open(const std::shared_ptr<Buffer>& buffer) {
  buffer_ = buffer;

  // Verify the buffer

  // Initiatilize the Flatbuffer interface
  table_ = static_cast<const void*>(fbs::GetCTable(buffer->data()));
  return true;
}

std::string Table::description() const {
  if (!has_description()) {
    return std::string("");
  }
  const fbs::CTable* table = static_cast<const fbs::CTable*>(table_);
  return table->description()->str();
}

bool Table::has_description() const {
  // null represented as 0 flatbuffer offset
  const fbs::CTable* table = static_cast<const fbs::CTable*>(table_);
  return table->description() !=  0;
}

int64_t Table::num_rows() const {
  const fbs::CTable* table = static_cast<const fbs::CTable*>(table_);
  return table->num_rows();
}

int Table::version() const {
  const fbs::CTable* table = static_cast<const fbs::CTable*>(table_);
  return table->version();
}

size_t Table::num_columns() const {
  const fbs::CTable* table = static_cast<const fbs::CTable*>(table_);
  return table->columns()->size();
}

std::shared_ptr<Column> Table::GetColumn(int i) const {
  const fbs::CTable* table = static_cast<const fbs::CTable*>(table_);
  const fbs::Column* col = table->columns()->Get(i);

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
  return std::shared_ptr<Column>();
}

std::shared_ptr<Column> Table::GetColumnNamed(const std::string& name) const {
  // Not yet implemented
  return std::shared_ptr<Column>();
}

// ----------------------------------------------------------------------
// Column

void FromFlatbuffer(const fbs::PrimitiveArray* values, ArrayMetadata& out) {
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

PrimitiveType::type Column::values_type() const {
  return values_.type;
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
  } else {
    result->metadata_.timezone = "";
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
