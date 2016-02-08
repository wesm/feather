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

#ifndef FEATHER_IO_H
#define FEATHER_IO_H

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace feather {

// Abstract output stream
class OutputStream {
 public:
  // Close the output stream
  virtual void Close() = 0;

  virtual size_t Tell() = 0;

  virtual void Write(const uint8_t* data, size_t length) = 0;
};


// An output stream that is an in-memory
class InMemoryOutputStream : public OutputStream {
 public:
  InMemoryOutputStream(size_t initial_capacity);

  virtual void Close() {}

  virtual size_t Tell();

  virtual void Write(const uint8_t* data, size_t length);

  // Hand off the in-memory data to a new owner
  void Transfer(std::vector<uint8_t>& out);

 private:
  uint8_t* Head();

  std::vector<uint8_t> buffer_;
  size_t size_;
  size_t capacity_;
};


// class FileOutputStream : public OutputStream {
//  public:

//  private:

// };

} // namespace feather

#endif // FEATHER_IO_H
