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

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "feather/exception.h"
#include "feather/io.h"
#include "feather/test-common.h"
#include "feather/reader.h"
#include "feather/writer.h"

using std::shared_ptr;
using std::unique_ptr;

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

  std::vector<uint8_t> output_;
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

PrimitiveArray MakePrimitiveArray(PrimitiveType::type type,
    int64_t length, int64_t null_count,
    const uint8_t* nulls, const uint8_t* values) {
  PrimitiveArray result;
  result.type = type;
  result.length = length;
  result.null_count = null_count;
  result.nulls = nulls;
  result.values = values;
  return result;
}


void AssertPrimitiveEquals(const PrimitiveArray& left,
    const PrimitiveArray& right) {
  EXPECT_EQ(left.type, right.type);
  EXPECT_EQ(left.encoding, right.encoding);
  EXPECT_EQ(left.offset, right.offset);
  EXPECT_EQ(left.length, right.length);
  EXPECT_EQ(left.null_count, right.null_count);
  EXPECT_EQ(left.total_bytes, right.total_bytes);
}


TEST_F(TestTableWriter, PrimitiveRoundTrip) {
  PrimitiveArray array;
}

TEST_F(TestTableWriter, VLenPrimitiveRoundTrip) {
  // UTF8 or BINARY
}

} // namespace feather
