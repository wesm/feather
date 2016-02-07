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
  void setUp() {
  }

  std::unique_ptr<File> FinishGetRoot() {
    fbuilder_.Finish();

    std::unique_ptr<File> result(new File());
    result->Open(fbuilder_.GetBuffer(), fbuilder_.BufferSize());

    return result;
  }

 protected:
  metadata::FileBuilder fbuilder_;
};

TEST_F(TestFileBuilder, EmptyTables) {
  std::unique_ptr<TableBuilder> table;

  table = fbuilder_.NewTable("a", 10);
  table->Finish();

  table = fbuilder_.NewTable("b", 20);
  table->Finish();

  table = fbuilder_.NewTable("c", 20);
  table->Finish();

  std::unique_ptr<File> file = FinishGetRoot();

  ASSERT_EQ(3, file->num_tables());
}

} // namespace metadata

} // namespace feather
