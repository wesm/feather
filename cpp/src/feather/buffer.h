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

#ifndef FEATHER_BUFFER_H
#define FEATHER_BUFFER_H

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace feather {

// ----------------------------------------------------------------------
// Buffer classes

// Immutable API for a chunk of bytes which may or may not be owned by the
// class instance
class Buffer : public std::enable_shared_from_this<Buffer> {
 public:
  Buffer(const uint8_t* data, int64_t size) :
      data_(data),
      size_(size) {}

  // An offset into data that is owned by another buffer, but we want to be
  // able to retain a valid pointer to it even after other shared_ptr's to the
  // parent buffer have been destroyed
  Buffer(const std::shared_ptr<Buffer>& parent, int64_t offset, int64_t size);

  std::shared_ptr<Buffer> get_shared_ptr() {
    return shared_from_this();
  }

  const uint8_t* data() const {
    return data_;
  }

  int64_t size() const {
    return size_;
  }

  // Returns true if this Buffer is referencing memory (possibly) owned by some
  // other buffer
  bool is_shared() const {
    return static_cast<bool>(parent_);
  }

  const std::shared_ptr<Buffer> parent() const {
    return parent_;
  }

 protected:
  const uint8_t* data_;
  int64_t size_;

  // nullptr by default, but may be set
  std::shared_ptr<Buffer> parent_;
};

// A Buffer whose contents can be mutated
class MutableBuffer : public Buffer {
 public:
  MutableBuffer(uint8_t* data, int64_t size) :
      Buffer(data, size) {
    mutable_data_ = data;
  }

  uint8_t* mutable_data() {
    return mutable_data_;
  }

  // Get a read-only view of this buffer
  std::shared_ptr<Buffer> GetImmutableView();

 protected:
  MutableBuffer() :
      Buffer(nullptr, 0),
      mutable_data_(nullptr) {}

  uint8_t* mutable_data_;
};

// A MutableBuffer whose memory is owned by the class instance. For example,
// for reading data out of files that you want to deallocate when this class is
// garbage-collected
class OwnedMutableBuffer : public MutableBuffer {
 public:
  explicit OwnedMutableBuffer(int64_t size);
  void Resize(int64_t new_size);

 private:
  std::vector<uint8_t> buffer_owner_;
};

} // namespace feather

#endif // FEATHER_BUFFER_H
