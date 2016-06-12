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

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "feather/common.h"
#include "feather/feather-c.h"
#include "feather/tests/test-common.h"
#include "feather/types.h"

#define ASSERT_CFEATHER_OK(s) do {              \
    feather_status _s = (s);                    \
    ASSERT_EQ(FEATHER_OK, _s);                  \
  } while (0);

namespace feather {

class TestCAPI : public ::testing::Test {
 public:
  virtual void TearDown() {
    for (const std::string& path : tmp_paths_) {
      try {
        std::remove(path.c_str());
      } catch (const std::exception& e) {
        (void) e;
      }
    }
  }

  void OpenWriter(const std::string& path) {
    const char* c_path = path.c_str();
    tmp_paths_.push_back(path);

    ASSERT_CFEATHER_OK(feather_writer_open_file(c_path, &writer_));
  }

  void CloseWriter() {
    ASSERT_CFEATHER_OK(feather_writer_close(writer_));
    ASSERT_CFEATHER_OK(feather_writer_free(writer_));
  }

  void OpenReader(const std::string& path) {
    ASSERT_CFEATHER_OK(feather_reader_open_file(path.c_str(), &reader_));
  }

  void CloseReader() {
    ASSERT_CFEATHER_OK(feather_reader_close(reader_));
    ASSERT_CFEATHER_OK(feather_reader_free(reader_));
  }

 protected:
  std::vector<std::string> tmp_paths_;

  feather_writer_t* writer_;
  feather_reader_t* reader_;
};

TEST_F(TestCAPI, FileNotExist) {
  feather_status s;
  feather_reader_t* reader;

  const char* path = "file-not-exist.feather";

  s = feather_reader_open_file(path, &reader);
  ASSERT_EQ(FEATHER_IO_ERROR, s);
}

TEST_F(TestCAPI, WriteNumRows) {
  const int num_rows = 1000;

  std::string path("test-cfeather-write-num-rows");

  OpenWriter(path);
  feather_writer_set_num_rows(writer_, num_rows);
  CloseWriter();

  OpenReader(path);
  ASSERT_EQ(num_rows, feather_reader_num_rows(reader_));
  ASSERT_EQ(0, feather_reader_num_columns(reader_));
  CloseReader();
}

void MakePrimitive(feather_type type,
    int64_t length, int64_t null_count,
    const uint8_t* nulls, const uint8_t* values,
    const int32_t* offsets, feather_array_t* out) {
  out->type = type;
  out->length = length;
  out->null_count = null_count;
  out->nulls = nulls;
  out->values = values;
  out->offsets = offsets;
}

static PrimitiveType::type ToFeatherType(feather_type ctype) {
  return static_cast<PrimitiveType::type>(static_cast<int>(ctype));
}

bool cfeather_array_equals(const feather_array_t* lhs, const feather_array_t* rhs) {
  if (lhs->type != rhs->type ||
      lhs->length != rhs->length ||
      lhs->null_count != rhs->null_count) {
    return false;
  }

  if (lhs->null_count > 0) {
    if (lhs->null_count != rhs->null_count ||
        memcmp(lhs->nulls, rhs->nulls, util::bytes_for_bits(lhs->length)) != 0) {
      return false;
    }
  }

  // TODO(wesm): variable-length dimensions
  // Fixed size, get the number of bytes from the length and value size
  if (memcmp(lhs->values, rhs->values,
          lhs->length * ByteSize(ToFeatherType(lhs->type))) != 0) {
    return false;
  }

  return true;
}

TEST_F(TestCAPI, PrimitiveRoundTrip) {
  int num_values = 1000;
  int num_nulls = 50;
  int64_t null_bytes = util::bytes_for_bits(num_values);

    // Generate some random data
  vector<uint8_t> null_buffer;
  vector<uint8_t> values_buffer;
  test::random_bytes(null_bytes, 0, &null_buffer);
  test::random_bytes(num_values * sizeof(int32_t), 0, &values_buffer);

  feather_array_t cvalues;
  feather_array_t cvalues_nn;

  MakePrimitive(FEATHER_INT32, num_values, num_nulls, &null_buffer[0],
      &values_buffer[0], nullptr, &cvalues);

  // A non-nullable version of this
  MakePrimitive(FEATHER_UINT32, num_values, 0, nullptr, &values_buffer[0],
      nullptr, &cvalues_nn);

  std::string path("test-cfeather-primitive-round-trip");

  OpenWriter(path);

  const char* name0 = "f0";
  const char* name1 = "f1";

  ASSERT_CFEATHER_OK(feather_writer_append_plain(writer_, name0, &cvalues));
  ASSERT_CFEATHER_OK(feather_writer_append_plain(writer_, name1, &cvalues_nn));

  CloseWriter();

  OpenReader(path);

  ASSERT_EQ(2, feather_reader_num_columns(reader_));

  feather_column_t col;
  ASSERT_CFEATHER_OK(feather_reader_get_column(reader_, 0, &col));
  ASSERT_EQ(FEATHER_COLUMN_PRIMITIVE, col.type);
  ASSERT_STREQ(name0, col.name);

  ASSERT_TRUE(cfeather_array_equals(&cvalues, &col.values));

  feather_column_free(&col);

  ASSERT_CFEATHER_OK(feather_reader_get_column(reader_, 1, &col));
  ASSERT_STREQ(name1, col.name);

  ASSERT_TRUE(cfeather_array_equals(&cvalues_nn, &col.values));

  feather_column_free(&col);

  CloseReader();
}

} // namespace feather
