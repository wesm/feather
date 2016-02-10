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

#include <gtest/gtest.h>

#include "feather/io.h"
#include "feather/test-common.h"

namespace feather {

TEST(TestBufferReader, Basics) {
  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::unique_ptr<BufferReader> reader(new BufferReader(&data[0], data.size()));

  ASSERT_EQ(0, reader->Tell());
  ASSERT_EQ(10, reader->size());

  size_t bytes_read;
  const uint8_t* buffer = reader->ReadNoCopy(4, &bytes_read);
  ASSERT_EQ(4, bytes_read);
  ASSERT_EQ(0, memcmp(buffer, &data[0], bytes_read));
  ASSERT_EQ(4, reader->Tell());

  buffer = reader->ReadNoCopy(10, &bytes_read);
  ASSERT_EQ(6, bytes_read);
  ASSERT_EQ(0, memcmp(buffer, &data[4], bytes_read));
  ASSERT_EQ(10, reader->Tell());
}

TEST(TestInMemoryOutputStream, Basics) {
  std::unique_ptr<InMemoryOutputStream> stream(new InMemoryOutputStream(8));

  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  stream->Write(&data[0], 4);
  ASSERT_EQ(4, stream->Tell());
  stream->Write(&data[4], data.size() - 4);

  std::vector<uint8_t> out;
  stream->Transfer(&out);

  assert_vector_equal(data, out);
}

} // namespace feather
