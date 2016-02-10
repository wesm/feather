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

#include "feather/reader.h"

#include <cstring>
#include <memory>

#include "feather/common.h"
#include "feather/exception.h"

namespace feather {


TableReader::TableReader(std::shared_ptr<RandomAccessReader> source) :
    source_(source) {
  size_t magic_size = strlen(FEATHER_MAGIC_BYTES);
  size_t footer_size = magic_size + sizeof(uint32_t);

  // Pathological issue where the file is smaller than
  if (source->size() < magic_size + footer_size) {
    throw FeatherException("File is too small to be a well-formed file");
  }

  size_t bytes_read;
  const uint8_t* buffer = source->ReadNoCopy(magic_size, &bytes_read);

  if (memcmp(buffer, FEATHER_MAGIC_BYTES, magic_size)) {
    throw FeatherException("Not a feather file");
  }

  // Now get the footer and verify
  source->Seek(source->size() - footer_size);
  buffer = source->ReadNoCopy(footer_size, &bytes_read);
  if (memcmp(buffer + sizeof(uint32_t), FEATHER_MAGIC_BYTES, magic_size)) {
    throw FeatherException("Feather file footer incomplete");
  }

  uint32_t metadata_length = *reinterpret_cast<const uint32_t*>(buffer);
  if (source->size() < magic_size + footer_size + metadata_length) {
    throw FeatherException("File is smaller than indicated metadata size");
  }
  source->Seek(source->size() - footer_size - metadata_length);
  buffer = source->ReadNoCopy(source->size() - footer_size - metadata_length,
      &bytes_read);

  if (!metadata_.Open(buffer, metadata_length)) {
    throw FeatherException("Invalid file metadata");
  }
}

bool TableReader::HasDescription() const {
  return metadata_.has_description();
}

std::string TableReader::GetDescription() const {
  return metadata_.description();
}

int64_t TableReader::num_rows() const {
  return metadata_.num_rows();
}

int64_t TableReader::num_columns() const {
  return metadata_.num_columns();
}

} // namespace feather
