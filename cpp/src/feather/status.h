// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A Status encapsulates the result of an operation.  It may indicate success,
// or it may indicate an error with an associated error message.
//
// Multiple threads can invoke const methods on a Status without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Status must use
// external synchronization.

// Adapted from Kudu github.com/cloudera/kudu

#ifndef FEATHER_STATUS_H_
#define FEATHER_STATUS_H_

#include <cstdint>
#include <cstring>
#include <string>

namespace feather {

#define RETURN_NOT_OK(s) do {                   \
    Status _s = (s);                            \
    if (!_s.ok()) return _s;                    \
  } while (0);

enum class StatusCode: char {
  OK = 0,
  OutOfMemory = 1,
  KeyError = 2,
  Invalid = 3,
  IOError = 4,

  NotImplemented = 10,
};

class Status {
 public:
  // Create a success status.
  Status() : state_(NULL) { }
  ~Status() { delete[] state_; }

  // Copy the specified status.
  Status(const Status& s);
  void operator=(const Status& s);

  // Return a success status.
  static Status OK() { return Status(); }

  // Return error status of an appropriate type.
  static Status OutOfMemory(const std::string& msg, int16_t posix_code = -1) {
    return Status(StatusCode::OutOfMemory, msg, posix_code);
  }

  static Status KeyError(const std::string& msg) {
    return Status(StatusCode::KeyError, msg, -1);
  }

  static Status IOError(const std::string& msg) {
    return Status(StatusCode::IOError, msg, -1);
  }

  static Status NotImplemented(const std::string& msg) {
    return Status(StatusCode::NotImplemented, msg, -1);
  }

  static Status Invalid(const std::string& msg) {
    return Status(StatusCode::Invalid, msg, -1);
  }

  // Returns true iff the status indicates success.
  bool ok() const { return (state_ == NULL); }

  bool IsOutOfMemory() const { return code() == StatusCode::OutOfMemory; }
  bool IsKeyError() const { return code() == StatusCode::KeyError; }
  bool IsIOError() const { return code() == StatusCode::IOError; }
  bool IsInvalid() const { return code() == StatusCode::Invalid; }
  bool IsNotImplemented() const { return code() == StatusCode::NotImplemented; }

  // Return a string representation of this status suitable for printing.
  // Returns the string "OK" for success.
  std::string ToString() const;

  // Return a string representation of the status code, without the message
  // text or posix code information.
  std::string CodeAsString() const;

  // Get the POSIX code associated with this Status, or -1 if there is none.
  int16_t posix_code() const;

 private:
  // OK status has a NULL state_.  Otherwise, state_ is a new[] array
  // of the following form:
  //    state_[0..3] == length of message
  //    state_[4]    == code
  //    state_[5..6] == posix_code
  //    state_[7..]  == message
  const char* state_;

  StatusCode code() const {
    return ((state_ == NULL) ?
        StatusCode::OK : static_cast<StatusCode>(state_[4]));
  }

  Status(StatusCode code, const std::string& msg, int16_t posix_code);
  static const char* CopyState(const char* s);
};

inline Status::Status(const Status& s) {
  state_ = (s.state_ == NULL) ? NULL : CopyState(s.state_);
}

inline void Status::operator=(const Status& s) {
  // The following condition catches both aliasing (when this == &s),
  // and the common case where both s and *this are ok.
  if (state_ != s.state_) {
    delete[] state_;
    state_ = (s.state_ == NULL) ? NULL : CopyState(s.state_);
  }
}

}  // namespace feather

#endif // FEATHER_STATUS_H_
