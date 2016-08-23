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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "feather/common.h"
#include "feather/metadata.h"

namespace feather {

namespace metadata {

class TestTableBuilder : public ::testing::Test {
 public:
  void SetUp() {
    tb_.reset(new TableBuilder(1000));
  }

  virtual void Finish() {
    tb_->Finish();

    table_.reset(new Table());
    table_->Open(tb_->GetBuffer());
  }

 protected:
  std::unique_ptr<TableBuilder> tb_;
  std::unique_ptr<Table> table_;
};


TEST_F(TestTableBuilder, Version) {
  Finish();
  ASSERT_EQ(kFeatherVersion, table_->version());
}

TEST_F(TestTableBuilder, EmptyTable) {
  Finish();

  ASSERT_FALSE(table_->has_description());
  ASSERT_EQ("", table_->description());
  ASSERT_EQ(1000, table_->num_rows());
  ASSERT_EQ(0, table_->num_columns());
}

TEST_F(TestTableBuilder, SetDescription) {
  std::string desc("this is some good data");
  tb_->SetDescription(desc);
  Finish();
  ASSERT_TRUE(table_->has_description());
  ASSERT_EQ(desc, table_->description());
}

void AssertArrayEquals(const ArrayMetadata& left, const ArrayMetadata& right) {
  EXPECT_EQ(left.type, right.type);
  EXPECT_EQ(left.encoding, right.encoding);
  EXPECT_EQ(left.offset, right.offset);
  EXPECT_EQ(left.length, right.length);
  EXPECT_EQ(left.null_count, right.null_count);
  EXPECT_EQ(left.total_bytes, right.total_bytes);
}


TEST_F(TestTableBuilder, AddPrimitiveColumn) {
  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("f0");

  ArrayMetadata values1;
  ArrayMetadata values2;
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
  ArrayMetadata values1(PrimitiveType::UINT8, Encoding::PLAIN,
      10000, 1000, 100, 4000);
  ArrayMetadata levels(PrimitiveType::UTF8, Encoding::PLAIN,
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
  ArrayMetadata values1(PrimitiveType::INT64, Encoding::PLAIN,
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

TEST_F(TestTableBuilder, AddDateColumn) {
  ArrayMetadata values1(PrimitiveType::INT64, Encoding::PLAIN,
      10000, 1000, 100, 4000);
  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("d0");
  cb->SetValues(values1);
  cb->SetDate();
  cb->Finish();

  Finish();

  auto col = table_->GetColumn(0);

  ASSERT_EQ(ColumnType::DATE, col->type());
  AssertArrayEquals(col->values(), values1);
}

TEST_F(TestTableBuilder, AddTimeColumn) {
  ArrayMetadata values1(PrimitiveType::INT64, Encoding::PLAIN,
      10000, 1000, 100, 4000);
  std::unique_ptr<ColumnBuilder> cb = tb_->AddColumn("c0");
  cb->SetValues(values1);
  cb->SetTime(TimeUnit::SECOND);
  cb->Finish();
  Finish();

  auto col = table_->GetColumn(0);

  ASSERT_EQ(ColumnType::TIME, col->type());
  AssertArrayEquals(col->values(), values1);

  const TimeColumn* t_ptr = static_cast<const TimeColumn*>(col.get());
  ASSERT_EQ(TimeUnit::SECOND, t_ptr->unit());
}

} // namespace metadata

} // namespace feather
