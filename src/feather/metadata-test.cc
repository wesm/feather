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

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "feather/metadata.h"

namespace feather {

namespace metadata {

class TestFileBuilder : public ::testing::Test {
 public:
  std::unique_ptr<File> Finish() {
    fbuilder_.Finish();

    std::unique_ptr<File> result(new File());
    result->Open(fbuilder_.GetBuffer(), fbuilder_.BufferSize());

    return result;
  }

 protected:
  metadata::FileBuilder fbuilder_;
};

TEST_F(TestFileBuilder, EmptyTableTests) {
  // Test adding a few tables without any columns
  std::unique_ptr<TableBuilder> tb;

  tb = fbuilder_.AddTable("a", 10);
  tb->Finish();

  tb = fbuilder_.AddTable("bb", 20);
  tb->Finish();

  tb = fbuilder_.AddTable("cccc", 1000000);
  tb->Finish();

  std::unique_ptr<File> file = Finish();
  ASSERT_EQ(3, file->num_tables());

  std::shared_ptr<Table> table;

  table = file->GetTable(0);
  ASSERT_EQ("a", table->name());
  ASSERT_EQ(10, table->num_rows());
  ASSERT_EQ(0, table->num_columns());

  table = file->GetTable(1);
  ASSERT_EQ("bb", table->name());
  ASSERT_EQ(20, table->num_rows());
  ASSERT_EQ(0, table->num_columns());

  table = file->GetTable(2);
  ASSERT_EQ("cccc", table->name());
  ASSERT_EQ(1000000, table->num_rows());
  ASSERT_EQ(0, table->num_columns());
}

// ----------------------------------------------------------------------
// Test column building

class TestTableBuilder : public TestFileBuilder {
 public:
  void SetUp() {
    tb_ = fbuilder_.AddTable("table", 1000);
  }

  virtual void Finish() {
    tb_->Finish();

    // Get the root now
    file_ = TestFileBuilder::Finish();

    // Get the table
    table_ = file_->GetTable(0);
  }

 protected:
  std::unique_ptr<TableBuilder> tb_;
  std::unique_ptr<File> file_;

  std::shared_ptr<Table> table_;
};

TEST_F(TestTableBuilder, AddPrimitiveColumn) {
  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("f0");

  PrimitiveArray values;
  values.type = PrimitiveType::INT32;
  values.encoding = Encoding::PLAIN;
  values.offset = 10000;
  values.length = 1000;
  values.null_count = 100;
  values.total_bytes = 4000;

  cb->SetValues(values);

  std::string user_meta = "as you wish";
  cb->SetUserMetadata(user_meta);

  cb->Finish();

  cb = tb_->AddColumn("f1");

  values.type = PrimitiveType::UTF8;
  values.encoding = Encoding::PLAIN;
  values.offset = 14000;
  values.length = 1000;
  values.null_count = 100;
  values.total_bytes = 10000;

  cb->SetValues(values);
  cb->Finish();

  Finish();

  auto col = table_->GetColumn(0);

  ASSERT_EQ("f0", col->name());
  ASSERT_EQ(ColumnType::PRIMITIVE, col->type());
  ASSERT_EQ(user_meta, col->user_metadata());

  auto rvalues = col->values();
  ASSERT_EQ(PrimitiveType::INT32, rvalues->type);
  ASSERT_EQ(Encoding::PLAIN, rvalues->encoding);
  ASSERT_EQ(10000, rvalues->offset);
  ASSERT_EQ(1000, rvalues->length);
  ASSERT_EQ(100, rvalues->null_count);
  ASSERT_EQ(4000, rvalues->total_bytes);

  col = table_->GetColumn(1);
  ASSERT_EQ("f1", col->name());
  ASSERT_EQ(ColumnType::PRIMITIVE, col->type());

  rvalues = col->values();
  ASSERT_EQ(PrimitiveType::UTF8, rvalues->type);
  ASSERT_EQ(Encoding::PLAIN, rvalues->encoding);
  ASSERT_EQ(14000, rvalues->offset);
  ASSERT_EQ(1000, rvalues->length);
  ASSERT_EQ(100, rvalues->null_count);
  ASSERT_EQ(10000, rvalues->total_bytes);
}

} // namespace metadata

} // namespace feather
