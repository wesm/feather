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

// Functions for NumPy / pandas conversion

#include <Python.h>

#include "feather/api.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include <numpy/arrayobject.h>

namespace feather {

namespace py {

#define DTYPE_CASE(TYPE)                                    \
  case NPY_##TYPE:                                          \
    {                                                       \
      FeatherSerializer<NPY_##TYPE> converter(arr, out);    \
      RETURN_NOT_OK(converter.Convert());                   \
    }                                                       \
    break;

template <int TYPE>
struct npy_traits {
};

template <>
struct npy_traits<NPY_BOOL> {
};

template <>
struct npy_traits<NPY_INT8> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::INT8;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_INT16> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::INT16;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_INT32> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::INT32;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_INT64> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::INT64;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_UINT8> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::UINT8;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_UINT16> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::UINT16;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_UINT32> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::UINT32;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_UINT64> {
  static constexpr PrimitiveType::type feather_type = PrimitiveType::UINT64;
  static constexpr bool supports_nulls = false;
};

template <>
struct npy_traits<NPY_FLOAT> {
  typedef float value_type;

  static constexpr PrimitiveType::type feather_type = PrimitiveType::FLOAT;
  static constexpr bool supports_nulls = true;

  static inline bool isnull(float v) {
    return v != v;
  }
};

template <>
struct npy_traits<NPY_DOUBLE> {
  typedef double value_type;

  static constexpr PrimitiveType::type feather_type = PrimitiveType::DOUBLE;
  static constexpr bool supports_nulls = false;

  static inline bool isnull(double v) {
    return v != v;
  }
};

template <>
struct npy_traits<NPY_OBJECT> {
  typedef PyObject* value_type;
  static constexpr bool supports_nulls = true;
};


template <int TYPE>
class FeatherSerializer {
 public:
  FeatherSerializer(PyArrayObject* arr, PrimitiveArray* out) :
      arr_(arr),
      out_(out) {}

  Status Convert() {
    out_->length = PyArray_SIZE(arr_);
    ConvertValues();
    return Status::OK();
  }

  int stride() const {
    return PyArray_STRIDES(arr_);
  }

  Status ConvertValues();

  bool is_strided() const {
    npy_intp* strides = PyArray_STRIDES(arr_);
    return strides[0] != PyArray_DESCR(arr_)->elsize;
  }

 private:
  PyArrayObject* arr_;
  PrimitiveArray* out_;
};

template <int TYPE>
inline Status FeatherSerializer<TYPE>::ConvertValues() {
  typedef npy_traits<TYPE> traits;
  out_->type = traits::feather_type;

  if (is_strided()) {
    return Status::Invalid("no support for strided data yet");
  }

  out_->values = static_cast<const uint8_t*>(PyArray_DATA(arr_));
  if (traits::supports_nulls) {
    int null_bytes = util::ceil_byte(out_->length);
    auto buffer = std::make_shared<OwnedMutableBuffer>(null_bytes);

    // TODO

    out_->nulls = buffer->data();
    out_->buffers.push_back(buffer);
  }
  return Status::OK();
}

template <>
inline Status FeatherSerializer<NPY_OBJECT>::ConvertValues() {
  return Status::Invalid("NYI");
}

Status pandas_to_primitive(PyObject* ao, PrimitiveArray* out) {
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(ao);

  if (arr->nd != 1) {
    return Status::Invalid("only handle 1-dimensional arrays");
  }

  switch(PyArray_DESCR(arr)->type_num) {
    DTYPE_CASE(INT8);
    DTYPE_CASE(INT16);
    DTYPE_CASE(INT32);
    DTYPE_CASE(INT64);
    DTYPE_CASE(UINT8);
    DTYPE_CASE(UINT16);
    DTYPE_CASE(UINT32);
    DTYPE_CASE(UINT64);
    // DTYPE_CASE(FLOAT);
    // DTYPE_CASE(DOUBLE);
    // DTYPE_CASE(OBJECT);
    default:
      std::stringstream ss;
      ss << "unsupported type " << PyArray_DESCR(arr)->type_num
         << std::endl;
      return Status::Invalid(ss.str());
  }
  return Status::OK();
}

} // namespace py

} // namespace feather
