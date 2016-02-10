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

namespace feather {

// ----------------------------------------------------------------------
// Input interfaces

// An abstract read-only file interface capable of performing Seeks (as opposed
// to being an input stream)
class RandomAccessReader {
 public:
  virtual ~RandomAccessReader() {}

  virtual size_t Tell() = 0;
  virtual void Seek(size_t pos) = 0;

  size_t size() {
    return size_;
  }

  // Most generic read method: *copies* bytes from the reader into a
  // preallocated memory region that you pass in.
  //
  // Other reader implementations may have a method that lets you get access to
  // data without copying. See BufferReader.
  //
  // @returns actual number of bytes read
  virtual size_t ReadInto(size_t nbytes, uint8_t* out) = 0;

 protected:
  size_t size_;
};


// File interface that interacts with a file on disk using operating-system
// level seek and read calls.
class LocalFileReader : public RandomAccessReader {
 public:
  virtual ~LocalFileReader();

  static std::unique_ptr<LocalFileReader> Open(const std::string& path);

  void CloseFile();

  virtual size_t Tell();
  virtual void Seek(size_t pos);

  // Returns actual number of bytes read
  virtual size_t ReadInto(size_t nbytes, uint8_t* out);

  bool is_open() const { return is_open_;}
  const std::string& path() const { return path_;}

 private:
  LocalFileReader(const std::string& path, size_t size, FILE* file) :
      path_(path),
      file_(file),
      is_open_(true) {
    size_ = size;
  }

  std::string path_;
  FILE* file_;
  bool is_open_;
};

// A file-like object that reads from virtual address space
class BufferReader : public RandomAccessReader {
 public:
  BufferReader(const uint8_t* buffer, size_t size) :
      buffer_(buffer),
      pos_(0) {
    size_ = size;
  }

  virtual size_t Tell();
  virtual void Seek(size_t pos);

  const uint8_t* ReadNoCopy(size_t nbytes, size_t* bytes_available);

  virtual size_t ReadInto(size_t nbytes, uint8_t* out);

 protected:
  const uint8_t* Head() {
    return buffer_ + pos_;
  }

  const uint8_t* buffer_;
  size_t pos_;
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
  void Transfer(std::vector<uint8_t>* out);

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
