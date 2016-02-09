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

namespace feather {

// ----------------------------------------------------------------------
// LocalFile methods

LocalFile::~LocalFile() {
  CloseFile();
}

void LocalFile::Open(const std::string& path) {
  path_ = path;
  file_ = fopen(path_.c_str(), "r");
  is_open_ = true;
}

void LocalFile::Close() {
  // Pure virtual
  CloseFile();
}

void LocalFile::CloseFile() {
  if (is_open_) {
    fclose(file_);
    is_open_ = false;
  }
}

size_t LocalFile::Size() {
  fseek(file_, 0L, SEEK_END);
  return Tell();
}

void LocalFile::Seek(size_t pos) {
  fseek(file_, pos, SEEK_SET);
}

size_t LocalFile::Tell() {
  return ftell(file_);
}

size_t LocalFile::Read(size_t nbytes, uint8_t* buffer) {
  return fread(buffer, 1, nbytes, file_);
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