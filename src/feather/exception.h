// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef FEATHER_EXCEPTION_H
#define FEATHER_EXCEPTION_H

#include <exception>
#include <sstream>
#include <string>

namespace feather {

class FeatherException : public std::exception {
 public:
  static void NYI(const std::string& msg) {
    std::stringstream ss;
    ss << "Not yet implemented: " << msg << ".";
    throw FeatherException(ss.str());
  }

  explicit FeatherException(const char* msg) : msg_(msg) {}
  explicit FeatherException(const std::string& msg) : msg_(msg) {}
  explicit FeatherException(const char* msg, exception& e) : msg_(msg) {}

  virtual ~FeatherException() throw() {}
  virtual const char* what() const throw() { return msg_.c_str(); }

 private:
  std::string msg_;
};

} // namespace feather

#endif // FEATHER_EXCEPTION_H
