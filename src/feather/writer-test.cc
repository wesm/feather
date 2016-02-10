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

#include "feather/io.h"
#include "feather/test-common.h"
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
  }

 protected:
  shared_ptr<InMemoryOutputStream> stream_;
  unique_ptr<TableWriter> writer_;

  std::vector<uint8_t> output_;
};

TEST_F(TestTableWriter, EmptyTable) {
}

} // namespace feather
