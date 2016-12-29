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

#ifndef FEATHER_COMPATIBILITY_H_
#define FEATHER_COMPATIBILITY_H_

// Compatibility for older versions of gcc without full C++11 support
#if defined(__GNUC__) && !defined(__clang__)

// gcc < 4.6
# if __GNUC__ == 4 && __GNUC_MINOR__ < 6

const class feather_nullptr_t {
public:
  template<class T> inline operator T*() const { return 0; }
private:
  void operator&() const; // NOLINT
} nullptr = {};
#  define nullptr_t feather_nullptr_t
# endif

// gcc <= 4.6
# if __GNUC__ == 4 && __GNUC_MINOR__ <= 6
#  define FEATHER_CPP0X_COMPATIBLE
#  define constexpr
#  define override
# endif

#endif

#endif /* FEATHER_COMPATIBILITY_H_ */
