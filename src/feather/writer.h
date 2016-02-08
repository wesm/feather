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

#ifndef FEATHER_WRITER_H
#define FEATHER_WRITER_H

#include "feather/io.h"
#include "feather/metadata.h"
#include "feather/types.h"

namespace feather {

struct PrimitiveData {
  PrimitiveType::type type;
  int64_t length;
  int64_t null_count;

  // If null_count == 0, treated as nullptr
  const uint8* nulls;

  const void* data;

  // For UTF8 and BINARY, not used otherwise
  const int32_t* offsets;
};

struct CategoryData {
  PrimitiveData indices;
  PrimitiveData levels;
  bool ordered;
}

struct DictionaryData {
  PrimitiveData dict_values;
  PrimitiveData indices;
};

class FileWriter;

class TableWriter {
 public:
  // Plain-encoded data
  void AppendPlain(const std::string& name, const PrimitiveData& values);

  // Dictionary-encoded primitive data. Especially useful for strings and
  // binary data
  void AppendDictEncoded(const std::string& name, const DictionaryData& data);

  // Category type data
  void AppendCategory(const std::string& name, const PrimitiveData& values,
      const PrimitiveData& levels, bool ordered = false);

  // Other primitive data types
  void AppendTimestamp(const std::string& name, const PrimitiveData& values
      const TimestampMetadata& meta);

 private:
  friend class FileWriter;

  FileWriter* parent_;
  std::unique_ptr<TableBuilder> metadata_;
};

class FileWriter {
 public:
  FileWriter(std::unique_ptr<OutputStream> stream);

  // Append a table to the metadata
  std::unique_ptr<TableWriter> AddTable(const std::string& name,
      int64_t num_rows);

  // We are done, write the file metadata and footer
  void Finalize();

 private:
  std::unique_ptr<OutputStream> stream_;
  metadata::FileBuilder metadata_;

  // Append a primitive array to the file
  void AppendPrimitiveArray();

  void AppendVariableArray();
};

} // namespace feather

#endif // FEATHER_WRITER_H
