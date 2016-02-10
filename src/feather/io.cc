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

#include "feather/io.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "feather/exception.h"

namespace feather {

// ----------------------------------------------------------------------
// BufferReader

size_t BufferReader::Tell() {
  return pos_;
}

void BufferReader::Seek(size_t pos) {
  if (pos >= size_) {
    std::stringstream ss;
    ss << "Cannot seek to " << pos
       << "File is length " << size_;
    throw FeatherException(ss.str());
  }
  pos_ = pos;
}

size_t BufferReader::ReadInto(size_t nbytes, uint8_t* out) {
  FeatherException::NYI("not implemented");
  return 0;
}

const uint8_t* BufferReader::ReadNoCopy(size_t nbytes, size_t* bytes_available) {
  *bytes_available = std::min(nbytes, size_ - pos_);
  const uint8_t* result = Head();
  pos_ += *bytes_available;
  return result;
}

// ----------------------------------------------------------------------
// LocalFileReader methods

LocalFileReader::~LocalFileReader() {
  CloseFile();
}

std::unique_ptr<LocalFileReader> LocalFileReader::Open(const std::string& path) {
  FILE* file = fopen(path.c_str(), "r");
  if (file == nullptr) {
    throw FeatherException("Unable to open file");
  }

  // Get and set file size
  fseek(file, 0L, SEEK_END);
  if (ferror(file)) {
    throw FeatherException("Unable to seek to end of file");
  }

  size_t size = ftell(file);

  auto result = std::unique_ptr<LocalFileReader>(
      new LocalFileReader(path, size, file));

  result->Seek(0);
  return result;
}

void LocalFileReader::CloseFile() {
  if (is_open_) {
    fclose(file_);
    is_open_ = false;
  }
}

void LocalFileReader::Seek(size_t pos) {
  fseek(file_, pos, SEEK_SET);
}

size_t LocalFileReader::Tell() {
  return ftell(file_);
}

size_t LocalFileReader::ReadInto(size_t nbytes, uint8_t* buffer) {
  size_t bytes_read = fread(buffer, 1, nbytes, file_);
  if (bytes_read < nbytes) {
    // Exception if not EOF
    if (!feof(file_)) {
      throw FeatherException("Error reading bytes from file");
    }
  }
  return bytes_read;
}

// ----------------------------------------------------------------------
// In-memory output stream

InMemoryOutputStream::InMemoryOutputStream(size_t initial_capacity) :
    size_(0),
    capacity_(initial_capacity) {
  if (initial_capacity == 0) {
    initial_capacity = 1024;
  }
  buffer_.resize(initial_capacity);
}

uint8_t* InMemoryOutputStream::Head() {
  return &buffer_[size_];
}

void InMemoryOutputStream::Write(const uint8_t* data, size_t length) {
  if (size_ + length > capacity_) {
    size_t new_capacity = capacity_ * 2;
    while (new_capacity < size_ + length) {
      new_capacity *= 2;
    }
    buffer_.resize(new_capacity);
    capacity_ = new_capacity;
  }
  memcpy(Head(), data, length);
  size_ += length;
}

size_t InMemoryOutputStream::Tell() {
  return size_;
}

void InMemoryOutputStream::Transfer(std::vector<uint8_t>& out) {
  buffer_.resize(size_);
  buffer_.swap(out);
  size_ = 0;
  capacity_ = buffer_.size();
}

} // namespace feather
