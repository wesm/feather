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
#include <random>
#include <vector>

using std::vector;

namespace feather {

#define ASSERT_RAISES(ENUM, expr)               \
  do {                                          \
    Status s = (expr);                          \
    ASSERT_TRUE(s.Is##ENUM());                  \
  } while (0)


#define ASSERT_OK(expr)                         \
  do {                                          \
    Status s = (expr);                          \
    ASSERT_TRUE(s.ok());                        \
  } while (0)


#define EXPECT_OK(expr)                         \
  do {                                          \
    Status s = (expr);                          \
    EXPECT_TRUE(s.ok());                        \
  } while (0)

namespace test {

template <typename T>
inline void assert_vector_equal(const vector<T>& left,
    const vector<T>& right) {
  ASSERT_EQ(left.size(), right.size());

  for (size_t i = 0; i < left.size(); ++i) {
    ASSERT_EQ(left[i], right[i]) << i;
  }
}

static inline void random_bytes(int64_t n, uint32_t seed, std::vector<uint8_t>* out) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> d(0, 255);

  for (int i = 0; i < n; ++i) {
    out->push_back(d(gen) & 0xFF);
  }
}

static inline void random_vlen_bytes(int64_t n, int max_value_size, uint32_t seed,
    std::vector<int32_t>* offsets, std::vector<uint8_t>* values) {
  std::mt19937 gen(seed);

  std::uniform_int_distribution<int32_t> len_dist(0, max_value_size);
  std::uniform_int_distribution<int> byte_dist(0, 255);

  int32_t offset = 0;
  for (int i = 0; i < n; ++i) {
    offsets->push_back(offset);

    int32_t length = len_dist(gen);

    // Generate bytes for the value in this slot
    for (int j = 0; j < length; ++j) {
      values->push_back(byte_dist(gen) & 0xFF);
    }
    offset += length;
  }
  // final (n + 1)-th offset
  offsets->push_back(offset);
}

} // namespace test

} // namespace feather
