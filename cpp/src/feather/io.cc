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

#ifdef __MINGW32__
#include "feather/mman.h"
#undef Realloc
#undef Free
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include <algorithm>
#include <cstring>
#include <sstream>

#include "feather/status.h"

namespace feather {

// ----------------------------------------------------------------------
// Buffer and its subclasses

// ----------------------------------------------------------------------
// BufferReader

Status RandomAccessReader::ReadAt(int64_t position, int64_t nbytes,
    std::shared_ptr<Buffer>* out) {
  // TODO(wesm): boundchecking
  RETURN_NOT_OK(Seek(position));
  return Read(nbytes, out);
}

int64_t BufferReader::Tell() const {
  return pos_;
}

Status BufferReader::Seek(int64_t pos) {
  if (pos < 0 || pos >= size_) {
    std::stringstream ss;
    ss << "Cannot seek to " << pos
       << "File is length " << size_;
    return Status::IOError(ss.str());
  }
  pos_ = pos;
  return Status::OK();
}

Status BufferReader::Read(int64_t nbytes, std::shared_ptr<Buffer>* out) {
  int64_t bytes_available = std::min(nbytes, size_ - pos_);
  *out = std::make_shared<Buffer>(Head(), bytes_available);
  pos_ += bytes_available;
  return Status::OK();
}

// ----------------------------------------------------------------------
// LocalFileReader methods

LocalFileReader::~LocalFileReader() {
  CloseFile();
}

Status LocalFileReader::Open(const std::string& path) {
  path_ = path;
  file_ = fopen(path.c_str(), "r");
  if (file_ == nullptr) {
    return Status::IOError("Unable to open file");
  }

  // Get and set file size
  fseek(file_, 0L, SEEK_END);
  if (ferror(file_)) {
    return Status::IOError("Unable to seek to end of file");
  }

  size_ = ftell(file_);
  RETURN_NOT_OK(Seek(0));

  is_open_ = true;

  return Status::OK();
}

void LocalFileReader::CloseFile() {
  if (is_open_) {
    fclose(file_);
    is_open_ = false;
  }
}

Status LocalFileReader::Seek(int64_t pos) {
  fseek(file_, pos, SEEK_SET);
  return Status::OK();
}

int64_t LocalFileReader::Tell() const {
  return ftell(file_);
}

Status LocalFileReader::Read(int64_t nbytes, std::shared_ptr<Buffer>* out) {
  auto buffer = std::make_shared<OwnedMutableBuffer>();
  RETURN_NOT_OK(buffer->Resize(nbytes));
  int64_t bytes_read = fread(buffer->mutable_data(), 1, nbytes, file_);
  if (bytes_read < nbytes) {
    // Exception if not EOF
    if (!feof(file_)) {
      return Status::IOError("Error reading bytes from file");
    }
    RETURN_NOT_OK(buffer->Resize(bytes_read));
  }
  *out = buffer;
  return Status::OK();
}

// ----------------------------------------------------------------------
// MemoryMapReader methods

MemoryMapReader::~MemoryMapReader() {
  CloseFile();
}

Status MemoryMapReader::Open(const std::string& path) {
  RETURN_NOT_OK(LocalFileReader::Open(path));
  data_ = reinterpret_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ,
          MAP_SHARED, fileno(file_), 0));
  if (data_ == nullptr) {
    return Status::IOError("Memory mapping file failed");
  }
  pos_  = 0;
  return Status::OK();
}

void MemoryMapReader::CloseFile() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }

  LocalFileReader::CloseFile();
}

Status MemoryMapReader::Seek(int64_t pos) {
  pos_ = pos;
  return Status::OK();
}

int64_t MemoryMapReader::Tell() const {
  return pos_;
}

Status MemoryMapReader::Read(int64_t nbytes, std::shared_ptr<Buffer>* out) {
  nbytes = std::min(nbytes, size_ - pos_);
  *out = std::make_shared<Buffer>(data_ + pos_, nbytes);
  return Status::OK();
}

// ----------------------------------------------------------------------
// In-memory output stream

InMemoryOutputStream::InMemoryOutputStream(int64_t initial_capacity) :
    size_(0),
    capacity_(initial_capacity) {
  if (initial_capacity == 0) {
    initial_capacity = 1024;
  }
  buffer_.reset(new OwnedMutableBuffer());
  buffer_->Resize(initial_capacity);
}

Status InMemoryOutputStream::Close() {
  return Status::OK();
}

uint8_t* InMemoryOutputStream::Head() {
  return buffer_->mutable_data() + size_;
}

Status InMemoryOutputStream::Write(const uint8_t* data, int64_t length) {
  if (size_ + length > capacity_) {
    int64_t new_capacity = capacity_ * 2;
    while (new_capacity < size_ + length) {
      new_capacity *= 2;
    }
    RETURN_NOT_OK(buffer_->Resize(new_capacity));
    capacity_ = new_capacity;
  }
  memcpy(Head(), data, length);
  size_ += length;
  return Status::OK();
}

int64_t InMemoryOutputStream::Tell() const {
  return size_;
}

std::shared_ptr<Buffer> InMemoryOutputStream::Finish() {
  buffer_->Resize(size_);
  std::shared_ptr<Buffer> result = buffer_;
  buffer_ = nullptr;

  // TODO(wesm): raise exceptions if user calls Write after Finish
  size_ = 0;
  capacity_ = 0;
  return result;
}

// ----------------------------------------------------------------------
// FileOutputStream

Status FileOutputStream::Open(const std::string& path) {
  path_ = path;
  file_ = fopen(path.c_str(), "wb");
  if (file_ == nullptr || ferror(file_)) {
    return Status::IOError("Unable to open file");
  }

  return Status::OK();
}

Status FileOutputStream::Close() {
  int code = fclose(file_);
  if (code != 0) {
    return Status::IOError("error closing file");
  }
  return Status::OK();
}

int64_t FileOutputStream::Tell() const {
  return ftell(file_);
}

Status FileOutputStream::Write(const uint8_t* data, int64_t length) {
  fwrite(data, 1, length, file_);
  if (ferror(file_)) {
    return Status::IOError("error writing bytes to file");
  }
  return Status::OK();
}

} // namespace feather
