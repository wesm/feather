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
#include <memory>

#include "feather/buffer.h"
#include "feather/io.h"
#include "feather/status.h"
#include "feather/tests/test-common.h"

namespace feather {

TEST(TestBufferReader, Basics) {
  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  auto data_buffer = std::make_shared<Buffer>(&data[0], data.size());
  std::unique_ptr<BufferReader> reader(new BufferReader(data_buffer));

  int64_t pos;
  ASSERT_OK(reader->Tell(&pos));

  ASSERT_EQ(0, pos);
  ASSERT_EQ(10, reader->size());

  std::shared_ptr<Buffer> buffer;
  ASSERT_OK(reader->Read(4, &buffer));
  ASSERT_EQ(4, buffer->size());
  ASSERT_EQ(0, memcmp(buffer->data(), &data[0], buffer->size()));

  ASSERT_OK(reader->Tell(&pos));
  ASSERT_EQ(4, pos);

  ASSERT_OK(reader->Read(10, &buffer));
  ASSERT_EQ(6, buffer->size());
  ASSERT_EQ(0, memcmp(buffer->data(), &data[4], buffer->size()));

  ASSERT_OK(reader->Tell(&pos));
  ASSERT_EQ(10, pos);
}

TEST(TestInMemoryOutputStream, Basics) {
  std::unique_ptr<InMemoryOutputStream> stream(new InMemoryOutputStream(8));

  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  ASSERT_OK(stream->Write(&data[0], 4));

  int64_t pos;
  ASSERT_OK(stream->Tell(&pos));

  ASSERT_EQ(4, pos);

  ASSERT_OK(stream->Write(&data[4], data.size() - 4));

  std::shared_ptr<Buffer> buffer = stream->Finish();
  ASSERT_EQ(0, memcmp(buffer->data(), &data[0], data.size()));
}

TEST(TestInMemoryOutputStream, WritePadded) {
  std::unique_ptr<InMemoryOutputStream> stream(new InMemoryOutputStream(8));
  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  int64_t bytes_written;
  ASSERT_OK(stream->WritePadded(data.data(), static_cast<int64_t>(data.size()),
          &bytes_written));

  ASSERT_EQ(16, bytes_written);

  int64_t pos;
  ASSERT_OK(stream->Tell(&pos));
  ASSERT_EQ(16, pos);

  const uint8_t padding_bytes[16] = {0};
  std::shared_ptr<Buffer> buffer = stream->Finish();

  ASSERT_EQ(16, buffer->size());
  ASSERT_EQ(0, memcmp(buffer->data() + data.size(), padding_bytes,
          buffer->size() - data.size()));
}

// TODO(wesm): These test are temporarily disabled on Windows because of the
// utf16-le mess
#if !defined (_MSC_VER)

TEST(LocalFileReader, NonExistentFile) {
  LocalFileReader reader;

  Status s = reader.Open("foo");
  ASSERT_FALSE(s.ok());
  ASSERT_TRUE(s.IsIOError());
}

TEST(FileOutputStream, NonExistentDirectory) {
  FileOutputStream writer;
  Status s = writer.Open("dir-does-not-exist/foo.feather");
  ASSERT_FALSE(s.ok());
  ASSERT_TRUE(s.IsIOError());
}

#endif

TEST(BufferBuilder, EmptyStrings) {
  BufferBuilder builder;

  builder.Append(nullptr, 0);
  builder.Append(nullptr, 0);
  builder.Append(nullptr, 0);

  std::shared_ptr<Buffer> result = builder.Finish();

  ASSERT_EQ(nullptr, result->data());
  ASSERT_EQ(0, result->size());
}

} // namespace feather
