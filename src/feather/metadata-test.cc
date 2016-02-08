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


void AssertArrayEquals(const PrimitiveArray& left, const PrimitiveArray& right) {
  EXPECT_EQ(left.type, right.type);
  EXPECT_EQ(left.encoding, right.encoding);
  EXPECT_EQ(left.offset, right.offset);
  EXPECT_EQ(left.length, right.length);
  EXPECT_EQ(left.null_count, right.null_count);
  EXPECT_EQ(left.total_bytes, right.total_bytes);
}


TEST_F(TestTableBuilder, AddPrimitiveColumn) {
  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("f0");

  PrimitiveArray values1;
  PrimitiveArray values2;
  values1.type = PrimitiveType::INT32;
  values1.encoding = Encoding::PLAIN;
  values1.offset = 10000;
  values1.length = 1000;
  values1.null_count = 100;
  values1.total_bytes = 4000;

  cb->SetValues(values1);

  std::string user_meta = "as you wish";
  cb->SetUserMetadata(user_meta);

  cb->Finish();

  cb = tb_->AddColumn("f1");

  values2.type = PrimitiveType::UTF8;
  values2.encoding = Encoding::PLAIN;
  values2.offset = 14000;
  values2.length = 1000;
  values2.null_count = 100;
  values2.total_bytes = 10000;

  cb->SetValues(values2);
  cb->Finish();

  Finish();

  ASSERT_EQ(2, table_->num_columns());

  auto col = table_->GetColumn(0);

  ASSERT_EQ("f0", col->name());
  ASSERT_EQ(ColumnType::PRIMITIVE, col->type());
  ASSERT_EQ(user_meta, col->user_metadata());

  AssertArrayEquals(col->values(), values1);

  col = table_->GetColumn(1);
  ASSERT_EQ("f1", col->name());
  ASSERT_EQ(ColumnType::PRIMITIVE, col->type());

  AssertArrayEquals(col->values(), values2);
}

TEST_F(TestTableBuilder, AddCategoryColumn) {
  PrimitiveArray values1(PrimitiveType::UINT8, Encoding::PLAIN,
      10000, 1000, 100, 4000);
  PrimitiveArray levels(PrimitiveType::UTF8, Encoding::PLAIN,
      14000, 10, 0, 300);

  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("c0");
  cb->SetValues(values1);
  cb->SetCategory(levels);
  cb->Finish();

  cb = tb_->AddColumn("c1");
  cb->SetValues(values1);
  cb->SetCategory(levels, true);
  cb->Finish();

  Finish();

  auto col = table_->GetColumn(0);

  ASSERT_EQ(ColumnType::CATEGORY, col->type());
  AssertArrayEquals(col->values(), values1);

  const CategoryColumn* cat_ptr = static_cast<const CategoryColumn*>(col.get());
  ASSERT_FALSE(cat_ptr->ordered());
  AssertArrayEquals(cat_ptr->levels(), levels);

  col = table_->GetColumn(1);
  cat_ptr = static_cast<const CategoryColumn*>(col.get());
  ASSERT_TRUE(cat_ptr->ordered());
  AssertArrayEquals(cat_ptr->levels(), levels);
}

TEST_F(TestTableBuilder, AddTimestampColumn) {
  PrimitiveArray values1(PrimitiveType::INT64, Encoding::PLAIN,
      10000, 1000, 100, 4000);
  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("c0");
  cb->SetValues(values1);
  cb->SetTimestamp(TimeUnit::MILLISECOND);
  cb->Finish();

  cb = tb_->AddColumn("c1");

  std::string tz("America/Los_Angeles");

  cb->SetValues(values1);
  cb->SetTimestamp(TimeUnit::SECOND, tz);
  cb->Finish();

  Finish();

  auto col = table_->GetColumn(0);

  ASSERT_EQ(ColumnType::TIMESTAMP, col->type());
  AssertArrayEquals(col->values(), values1);

  const TimestampColumn* ts_ptr = static_cast<const TimestampColumn*>(col.get());
  ASSERT_EQ(TimeUnit::MILLISECOND, ts_ptr->unit());

  col = table_->GetColumn(1);
  ts_ptr = static_cast<const TimestampColumn*>(col.get());
  ASSERT_EQ(TimeUnit::SECOND, ts_ptr->unit());
  ASSERT_EQ(tz, ts_ptr->timezone());
}

} // namespace metadata

} // namespace feather
