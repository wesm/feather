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
#include "feather/exception.h"
#include "feather/io.h"
#include "feather/reader.h"
#include "feather/test-common.h"
#include "feather/writer.h"

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

namespace feather {

class TestTableWriter : public ::testing::Test {
 public:
  void SetUp() {
    stream_ = std::make_shared<InMemoryOutputStream>(1024);
    writer_.reset(new TableWriter(stream_));
  }

  void Finish() {
    // Write table footer
    writer_->Finalize();
    stream_->Transfer(&output_);

    shared_ptr<BufferReader> buffer(new BufferReader(&output_[0], output_.size()));
    reader_.reset(new TableReader(buffer));
  }

 protected:
  shared_ptr<InMemoryOutputStream> stream_;
  unique_ptr<TableWriter> writer_;
  unique_ptr<TableReader> reader_;

  vector<uint8_t> output_;
};

TEST_F(TestTableWriter, EmptyTable) {
  Finish();

  ASSERT_FALSE(reader_->HasDescription());
  ASSERT_THROW(reader_->GetDescription(), FeatherException);

  ASSERT_EQ(0, reader_->num_rows());
  ASSERT_EQ(0, reader_->num_columns());
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

PrimitiveArray MakeFixedSize(PrimitiveType::type type,
    int64_t length, int64_t null_count,
    const uint8_t* nulls, const uint8_t* values) {
  PrimitiveArray result;
  result.type = type;
  result.length = length;
  result.null_count = null_count;
  result.nulls = nulls;
  result.values = values;
  result.offsets = nullptr;
  return result;
}

TEST_F(TestTableWriter, PrimitiveRoundTrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int null_bytes = util::ceil_byte(num_values);

  // Generate some random data
  vector<uint8_t> null_buffer(null_bytes);
  vector<uint8_t> values_buffer(num_values * sizeof(int32_t));

  test::random_bytes(null_buffer.size(), 0, &null_buffer);
  test::random_bytes(values_buffer.size(), 0, &values_buffer);

  PrimitiveArray array = MakeFixedSize(PrimitiveType::INT32, num_values,
      num_nulls, &null_buffer[0], &values_buffer[0]);

  writer_->AppendPlain("f0", array);
  Finish();

  auto col = reader_->GetColumn(0);
  ASSERT_TRUE(col->values().Equals(array));
}

TEST_F(TestTableWriter, VLenPrimitiveRoundTrip) {
  // UTF8 or BINARY
}

} // namespace feather
