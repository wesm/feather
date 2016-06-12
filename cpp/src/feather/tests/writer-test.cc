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

#include <memory>
#include <vector>

#include "feather/common.h"
#include "feather/io.h"
#include "feather/reader.h"
#include "feather/status.h"
#include "feather/writer.h"
#include "feather/tests/test-common.h"

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace feather {

class TestTableWriter : public ::testing::Test {
 public:
  void SetUp() {
    stream_ = std::make_shared<InMemoryOutputStream>(1024);
    writer_.reset(new TableWriter());
    ASSERT_OK(writer_->Open(stream_));
  }

  void Finish() {
    // Write table footer
    ASSERT_OK(writer_->Finalize());

    output_ = stream_->Finish();

    shared_ptr<BufferReader> buffer(new BufferReader(output_));
    reader_.reset(new TableReader());
    ASSERT_OK(reader_->Open(buffer));
  }

 protected:
  shared_ptr<InMemoryOutputStream> stream_;
  unique_ptr<TableWriter> writer_;
  unique_ptr<TableReader> reader_;

  std::shared_ptr<Buffer> output_;
};

TEST_F(TestTableWriter, EmptyTable) {
  Finish();

  ASSERT_FALSE(reader_->HasDescription());
  ASSERT_EQ("", reader_->GetDescription());

  ASSERT_EQ(0, reader_->num_rows());
  ASSERT_EQ(0, reader_->num_columns());
}

TEST_F(TestTableWriter, SetNumRows) {
  writer_->SetNumRows(1000);
  Finish();
  ASSERT_EQ(1000, reader_->num_rows());
}

TEST_F(TestTableWriter, SetDescription) {
  std::string desc("contents of the file");
  writer_->SetDescription(desc);
  Finish();

  ASSERT_TRUE(reader_->HasDescription());
  ASSERT_EQ(desc, reader_->GetDescription());

  ASSERT_EQ(0, reader_->num_rows());
  ASSERT_EQ(0, reader_->num_columns());
}

PrimitiveArray MakePrimitive(PrimitiveType::type type,
    int64_t length, int64_t null_count,
    const uint8_t* nulls, const uint8_t* values,
    const int32_t* offsets) {
  PrimitiveArray result;
  result.type = type;
  result.length = length;
  result.null_count = null_count;
  result.nulls = nulls;
  result.values = values;
  result.offsets = offsets;
  return result;
}

TEST_F(TestTableWriter, PrimitiveRoundTrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<uint8_t> values_buffer;
  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_bytes(num_values * sizeof(int32_t), 0, &values_buffer);

  PrimitiveArray array = MakePrimitive(PrimitiveType::INT32, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0], nullptr);

  // A non-nullable version of this
  PrimitiveArray nn_array = MakePrimitive(PrimitiveType::INT32, num_values,
      0, nullptr, &values_buffer[0], nullptr);

  ASSERT_OK(writer_->AppendPlain("f0", array));
  ASSERT_OK(writer_->AppendPlain("f1", nn_array));
  Finish();

  std::unique_ptr<Column> col;
  ASSERT_OK(reader_->GetColumn(0, &col));
  ASSERT_TRUE(col->values().Equals(array));
  ASSERT_EQ("f0", col->metadata()->name());

  ASSERT_OK(reader_->GetColumn(1, &col));
  ASSERT_TRUE(col->values().Equals(nn_array));
  ASSERT_EQ("f1", col->metadata()->name());
}

TEST_F(TestTableWriter, CategoryRoundtrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int num_levels = 10;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<uint8_t> values_buffer;
  vector<uint8_t> levels_buffer;

  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_bytes(num_values * sizeof(int32_t), 0, &values_buffer);
  test::random_bytes(num_levels * sizeof(uint8_t), 0, &levels_buffer);

  PrimitiveArray values = MakePrimitive(PrimitiveType::INT32, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0], nullptr);
  PrimitiveArray levels = MakePrimitive(PrimitiveType::INT8, num_levels,
      0, nullptr, &levels_buffer[0], nullptr);

  ASSERT_OK(writer_->AppendCategory("f0", values, levels, true));
  Finish();

  std::unique_ptr<Column> col;
  ASSERT_OK(reader_->GetColumn(0, &col));
  ASSERT_EQ("f0", col->metadata()->name());
  ASSERT_EQ(ColumnType::CATEGORY, col->type());
  ASSERT_TRUE(col->values().Equals(values));

  auto cat_col = static_cast<const CategoryColumn*>(col.get());
  ASSERT_TRUE(cat_col->levels().Equals(levels));
  ASSERT_TRUE(cat_col->ordered());
}

TEST_F(TestTableWriter, TimestampRoundtrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<uint8_t> values_buffer;
  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_bytes(num_values * sizeof(int64_t), 0, &values_buffer);

  PrimitiveArray values = MakePrimitive(PrimitiveType::INT64, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0], nullptr);

  TimestampMetadata metadata;
  metadata.unit = TimeUnit::SECOND;
  metadata.timezone = std::string("America/Los_Angeles");

  ASSERT_OK(writer_->AppendTimestamp("f0", values, metadata));
  Finish();

  std::unique_ptr<Column> col;
  ASSERT_OK(reader_->GetColumn(0, &col));
  ASSERT_EQ("f0", col->metadata()->name());
  ASSERT_EQ(ColumnType::TIMESTAMP, col->type());
  ASSERT_TRUE(col->values().Equals(values));

  auto ts_col = static_cast<const TimestampColumn*>(col.get());
  ASSERT_EQ(metadata.unit, ts_col->unit());
  ASSERT_EQ(metadata.timezone, ts_col->timezone());
}

TEST_F(TestTableWriter, DateRoundtrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<uint8_t> values_buffer;
  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_bytes(num_values * sizeof(int64_t), 0, &values_buffer);

  PrimitiveArray values = MakePrimitive(PrimitiveType::INT32, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0], nullptr);

  ASSERT_OK(writer_->AppendDate("f0", values));
  Finish();

  std::unique_ptr<Column> col;
  ASSERT_OK(reader_->GetColumn(0, &col));
  ASSERT_EQ("f0", col->metadata()->name());
  ASSERT_EQ(ColumnType::DATE, col->type());
  ASSERT_TRUE(col->values().Equals(values));
}

TEST_F(TestTableWriter, TimeRoundtrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<uint8_t> values_buffer;
  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_bytes(num_values * sizeof(int64_t), 0, &values_buffer);

  PrimitiveArray values = MakePrimitive(PrimitiveType::INT64, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0], nullptr);

  TimeMetadata metadata;
  metadata.unit = TimeUnit::SECOND;
  ASSERT_OK(writer_->AppendTime("f0", values, metadata));
  Finish();

  std::unique_ptr<Column> col;
  ASSERT_OK(reader_->GetColumn(0, &col));
  ASSERT_EQ("f0", col->metadata()->name());
  ASSERT_EQ(ColumnType::TIME, col->type());
  ASSERT_TRUE(col->values().Equals(values));

  auto ts_col = static_cast<const TimeColumn*>(col.get());
  ASSERT_EQ(metadata.unit, ts_col->unit());
}

TEST_F(TestTableWriter, VLenPrimitiveRoundTrip) {
  // UTF8 or BINARY
  int num_values = 1000;
  int num_nulls = 50;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<int32_t> offsets_buffer;
  vector<uint8_t> values_buffer;

  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_vlen_bytes(num_values, 10, 0, &offsets_buffer, &values_buffer);

  PrimitiveArray array = MakePrimitive(PrimitiveType::UTF8, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0], &offsets_buffer[0]);

  // A non-nullable version
  PrimitiveArray nn_array = MakePrimitive(PrimitiveType::UTF8, num_values,
      0, nullptr, &values_buffer[0], &offsets_buffer[0]);

  ASSERT_OK(writer_->AppendPlain("f0", array));
  ASSERT_OK(writer_->AppendPlain("f1", nn_array));
  Finish();

  std::unique_ptr<Column> col;
  ASSERT_OK(reader_->GetColumn(0, &col));
  ASSERT_TRUE(col->values().Equals(array));
  ASSERT_EQ("f0", col->metadata()->name());

  ASSERT_OK(reader_->GetColumn(1, &col));
  ASSERT_TRUE(col->values().Equals(nn_array));
  ASSERT_EQ("f1", col->metadata()->name());
}

} // namespace feather
