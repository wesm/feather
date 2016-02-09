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
#include <stdio.h>
#include <string>
#include <vector>

namespace feather {

// ----------------------------------------------------------------------
// Input interfaces

// An abstract read-only file interface
class FileLike {
 public:
  virtual ~FileLike() {}

  virtual void Close() = 0;
  virtual size_t Size() = 0;
  virtual size_t Tell() = 0;
  virtual void Seek(size_t pos) = 0;

  // Returns actual number of bytes read
  virtual size_t Read(size_t nbytes, uint8_t* out) = 0;
};


// File interface that interacts with a file on disk
class LocalFile : public FileLike {
 public:
  LocalFile() : file_(nullptr), is_open_(false) {}
  virtual ~LocalFile();

  void Open(const std::string& path);

  virtual void Close();
  virtual size_t Size();
  virtual size_t Tell();
  virtual void Seek(size_t pos);

  // Returns actual number of bytes read
  virtual size_t Read(size_t nbytes, uint8_t* out);

  bool is_open() const { return is_open_;}
  const std::string& path() const { return path_;}

 private:
  void CloseFile();

  std::string path_;
  FILE* file_;
  bool is_open_;
};

// ----------------------------------------------------------------------
// Output interfaces

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
  explicit InMemoryOutputStream(size_t initial_capacity);

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
