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
#include <string>

#include "feather/compatibility.h"

namespace feather {

class Buffer;
class OwnedMutableBuffer;
class Status;

class FileInterface;

// ----------------------------------------------------------------------
// Input interfaces

// An abstract read-only file interface capable of performing Seeks (as opposed
// to being an input stream)
class RandomAccessReader {
 public:
  virtual ~RandomAccessReader() {}

  virtual Status Tell(int64_t* pos) const = 0;
  virtual Status Seek(int64_t pos) = 0;

  // Read data from source at position (seeking if necessary), returning copy
  // only if necessary. Lifetime of read data is managed by the returned Buffer
  // instance
  Status ReadAt(int64_t position, int64_t nbytes, std::shared_ptr<Buffer>* out);

  // Read bytes from source at current position
  virtual Status Read(int64_t nbytes, std::shared_ptr<Buffer>* out) = 0;

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
  LocalFileReader();

  virtual ~LocalFileReader();

  Status Open(const std::string& path);
  void CloseFile();

  Status Tell(int64_t* pos) const override;
  Status Seek(int64_t pos) override;
  Status Read(int64_t nbytes, std::shared_ptr<Buffer>* out) override;

 protected:
  std::unique_ptr<FileInterface> impl_;
};

class MemoryMapReader : public LocalFileReader {
 public:
  MemoryMapReader() :
      LocalFileReader(),
      data_(nullptr),
      pos_(0) {}

  virtual ~MemoryMapReader();

  Status Open(const std::string& path);
  void CloseFile();

  Status Tell(int64_t* pos) const override;
  Status Seek(int64_t pos) override;
  Status Read(int64_t nbytes, std::shared_ptr<Buffer>* out) override;

 private:
  uint8_t* data_;
  int64_t pos_;
};

// ----------------------------------------------------------------------
// A file-like object that reads from virtual address space
class BufferReader : public RandomAccessReader {
 public:
  explicit BufferReader(const std::shared_ptr<Buffer>& buffer);

  Status Tell(int64_t* pos) const override;
  Status Seek(int64_t pos) override;
  Status Read(int64_t nbytes, std::shared_ptr<Buffer>* out) override;

 protected:
  const uint8_t* Head() {
    return data_ + pos_;
  }

  std::shared_ptr<Buffer> buffer_;
  const uint8_t* data_;
  int64_t pos_;
};

// ----------------------------------------------------------------------
// Output interfaces

// Abstract output stream
class OutputStream {
 public:
  virtual ~OutputStream() {}

  // Close the output stream
  virtual Status Close() = 0;

  virtual Status Tell(int64_t* pos) const = 0;

  virtual Status Write(const uint8_t* data, int64_t length) = 0;

  // Call Write and add additional padding bytes if length is not a multiple of
  // the alignment parameter
  Status WritePadded(const uint8_t* data, int64_t length, int64_t* bytes_written);
};


// An output stream that is an in-memory
class InMemoryOutputStream : public OutputStream {
 public:
  explicit InMemoryOutputStream(int64_t initial_capacity);

  virtual ~InMemoryOutputStream() {}

  Status Close() override;
  Status Tell(int64_t* pos) const override;
  Status Write(const uint8_t* data, int64_t length) override;

  // Hand off the buffered data to a new owner
  std::shared_ptr<Buffer> Finish();

 private:
  uint8_t* Head();

  std::shared_ptr<OwnedMutableBuffer> buffer_;
  int64_t size_;
  int64_t capacity_;
};

class FileOutputStream : public OutputStream {
 public:
  FileOutputStream();
  ~FileOutputStream();

  Status Open(const std::string& path);

  Status Close() override;
  Status Tell(int64_t* pos) const override;
  Status Write(const uint8_t* data, int64_t length) override;

  // Hand off the buffered data to a new owner
  std::shared_ptr<Buffer> Finish();

 private:
  std::unique_ptr<FileInterface> impl_;
};

} // namespace feather

#endif // FEATHER_IO_H
