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

#ifndef FEATHER_COMMON_H
#define FEATHER_COMMON_H

namespace feather {

static constexpr const char* FEATHER_MAGIC_BYTES = "FEA1";

namespace util {

static inline size_t ceil_byte(size_t size) {
  return (size + 7) & ~7;
}

} // namespace util

} // namespace feather

#endif // FEATHER_COMMON_H
