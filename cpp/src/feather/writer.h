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

#include <memory>
#include <string>

#include "feather/io.h"
#include "feather/metadata.h"
#include "feather/types.h"

namespace feather {

class TableWriter {
 public:
  TableWriter();

  Status Open(const std::shared_ptr<OutputStream>& stream);
  static Status OpenFile(const std::string& abspath,
      std::unique_ptr<TableWriter>* out);

  void SetDescription(const std::string& desc);
  void SetNumRows(int64_t num_rows);

  // Plain-encoded data
  Status AppendPlain(const std::string& name, const PrimitiveArray& values);

  // Dictionary-encoded primitive data. Especially useful for strings and
  // binary data
  void AppendDictEncoded(const std::string& name, const DictEncodedArray& data);

  // Category type data
  Status AppendCategory(const std::string& name, const PrimitiveArray& values,
      const PrimitiveArray& levels, bool ordered = false);

  // Other primitive data types
  Status AppendTimestamp(const std::string& name, const PrimitiveArray& values,
      const TimestampMetadata& meta);

  Status AppendDate(const std::string& name, const PrimitiveArray& values);

  Status AppendTime(const std::string& name, const PrimitiveArray& values,
      const TimeMetadata& meta);

  // We are done, write the file metadata and footer
  Status Finalize();

 private:
  Status Init();

  std::shared_ptr<OutputStream> stream_;

  bool initialized_stream_;
  metadata::TableBuilder metadata_;

  Status AppendPrimitive(const PrimitiveArray& values, ArrayMetadata* out);
};

} // namespace feather

#endif // FEATHER_WRITER_H
