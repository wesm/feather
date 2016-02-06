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

namespace feather {

namespace metadata {

// ----------------------------------------------------------------------
// FileBuilder

void FileBuilder::AddTable(const TableBuilder& table) {
  auto fb_table = fbs::CreateCTable(fbb_, fbb_.CreateString(table.name_),
      table.num_rows_,
      fbb_.CreateVector(table.columns_));
  tables_.push_back(fb_table);
}

// ---------------------------------------------------------------------
// File

bool File::Open(const void* buffer, size_t length) {
  // Verify the buffer

  // Initiatilize the Flatbuffer interface
  file_ = fbs::GetFile(buffer);
  return true;
}

// ----------------------------------------------------------------------
// Table

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

// ----------------------------------------------------------------------
// Column

Column::Column(const fbs::Column* col) {
  column_ = col;
}

} // namespace feather

} // namespace feather
