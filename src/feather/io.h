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
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#include "feather/buffer.h"

namespace feather {

// ----------------------------------------------------------------------
// Input interfaces

// An abstract read-only file interface capable of performing Seeks (as opposed
// to being an input stream)
class RandomAccessReader {
 public:
  virtual ~RandomAccessReader() {}

  virtual int64_t Tell() const = 0;
  virtual void Seek(int64_t pos) = 0;

  // Read data from source at position (seeking if necessary), returning copy
  // only if necessary. Lifetime of read data is managed by the returned Buffer
  // instance
  //
  // @returns: shared_ptr<Buffer>
  std::shared_ptr<Buffer> ReadAt(int64_t position, int64_t nbytes);

  // Read bytes from source at current position
  virtual std::shared_ptr<Buffer> Read(int64_t nbytes) = 0;

  int64_t size() {
    return size_;
  }

 protected:
  int64_t size_;
};


// File interface that interacts with a file on disk using operating-system
// level seek and read calls.
class LocalFileReader : public RandomAccessReader {
 public:
  virtual ~LocalFileReader();

  static std::unique_ptr<LocalFileReader> Open(const std::string& path);

  void CloseFile();

  virtual int64_t Tell() const;
  virtual void Seek(int64_t pos);

  virtual std::shared_ptr<Buffer> Read(int64_t nbytes);

  bool is_open() const { return is_open_;}
  const std::string& path() const { return path_;}

 private:
  LocalFileReader(const std::string& path, int64_t size, FILE* file) :
      path_(path),
      file_(file),
      is_open_(true) {
    size_ = size;
  }

  std::string path_;
  FILE* file_;
  bool is_open_;
};

// ----------------------------------------------------------------------
// A file-like object that reads from virtual address space
class BufferReader : public RandomAccessReader {
 public:
  BufferReader(const uint8_t* buffer, int64_t size) :
      buffer_(buffer),
      pos_(0) {
    size_ = size;
  }

  virtual int64_t Tell() const;
  virtual void Seek(int64_t pos);

  virtual std::shared_ptr<Buffer> Read(int64_t nbytes);

 protected:
  const uint8_t* Head() {
    return buffer_ + pos_;
  }

  const uint8_t* buffer_;
  int64_t pos_;
};

// A file reader that uses a memory-mapped file as its internal buffer rather
// than issuing operating system file commands
class MemoryMapReader : public BufferReader {
  MemoryMapReader();

  void Open(const std::string* path);
};

// ----------------------------------------------------------------------
// Output interfaces

// Abstract output stream
class OutputStream {
 public:
  // Close the output stream
  virtual void Close() = 0;

  virtual int64_t Tell() const = 0;

  virtual void Write(const uint8_t* data, int64_t length) = 0;
};


// An output stream that is an in-memory
class InMemoryOutputStream : public OutputStream {
 public:
  explicit InMemoryOutputStream(int64_t initial_capacity);

  virtual void Close() {}

  virtual int64_t Tell() const;

  virtual void Write(const uint8_t* data, int64_t length);

  // Hand off the in-memory data to a new owner
  void Transfer(std::vector<uint8_t>* out);

 private:
  uint8_t* Head();

  std::vector<uint8_t> buffer_;
  int64_t size_;
  int64_t capacity_;
};

} // namespace feather

#endif // FEATHER_IO_H
