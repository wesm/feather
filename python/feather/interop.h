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

// ----------------------------------------------------------------------
// Serialization

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
struct npy_traits<NPY_FLOAT32> {
  typedef float value_type;

  static constexpr PrimitiveType::type feather_type = PrimitiveType::FLOAT;
  static constexpr bool supports_nulls = true;

  static inline bool isnull(float v) {
    return v != v;
  }
};

template <>
struct npy_traits<NPY_FLOAT64> {
  typedef double value_type;

  static constexpr PrimitiveType::type feather_type = PrimitiveType::DOUBLE;
  static constexpr bool supports_nulls = true;

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
    // int null_bytes = util::ceil_byte(out_->length);
    // auto buffer = std::make_shared<OwnedMutableBuffer>(null_bytes);

    // // TODO

    // out_->nulls = buffer->data();
    // out_->buffers.push_back(buffer);
    out_->null_count = 0;
  } else {
    out_->null_count = 0;
  }
  return Status::OK();
}

template <>
inline Status FeatherSerializer<NPY_OBJECT>::ConvertValues() {
  return Status::Invalid("NYI");
}

#define TO_FEATHER_CASE(TYPE)                               \
  case NPY_##TYPE:                                          \
    {                                                       \
      FeatherSerializer<NPY_##TYPE> converter(arr, out);    \
      RETURN_NOT_OK(converter.Convert());                   \
    }                                                       \
    break;

Status pandas_to_primitive(PyObject* ao, PrimitiveArray* out) {
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(ao);

  if (arr->nd != 1) {
    return Status::Invalid("only handle 1-dimensional arrays");
  }

  switch(PyArray_DESCR(arr)->type_num) {
    TO_FEATHER_CASE(INT8);
    TO_FEATHER_CASE(INT16);
    TO_FEATHER_CASE(INT32);
    TO_FEATHER_CASE(INT64);
    TO_FEATHER_CASE(UINT8);
    TO_FEATHER_CASE(UINT16);
    TO_FEATHER_CASE(UINT32);
    TO_FEATHER_CASE(UINT64);
    TO_FEATHER_CASE(FLOAT);
    TO_FEATHER_CASE(DOUBLE);
    // TO_FEATHER_CASE(OBJECT);
    default:
      std::stringstream ss;
      ss << "unsupported type " << PyArray_DESCR(arr)->type_num
         << std::endl;
      return Status::Invalid(ss.str());
  }
  return Status::OK();
}

// ----------------------------------------------------------------------
// Deserialization

template <int TYPE>
struct feather_traits {
};

template <>
struct feather_traits<PrimitiveType::BOOL> {
};

template <>
struct feather_traits<PrimitiveType::INT8> {
  static constexpr int npy_type = NPY_INT8;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::INT16> {
  static constexpr int npy_type = NPY_INT16;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::INT32> {
  static constexpr int npy_type = NPY_INT32;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::INT64> {
  static constexpr int npy_type = NPY_INT64;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::UINT8> {
  static constexpr int npy_type = NPY_UINT8;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::UINT16> {
  static constexpr int npy_type = NPY_UINT16;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::UINT32> {
  static constexpr int npy_type = NPY_UINT32;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::UINT64> {
  static constexpr int npy_type = NPY_UINT64;
  static constexpr bool supports_nulls = false;
};

template <>
struct feather_traits<PrimitiveType::FLOAT> {
  static constexpr int npy_type = NPY_FLOAT32;
  static constexpr bool supports_nulls = true;
};

template <>
struct feather_traits<PrimitiveType::DOUBLE> {
  static constexpr int npy_type = NPY_FLOAT64;
  static constexpr bool supports_nulls = true;
};


template <int TYPE>
class FeatherDeserializer {
 public:
  FeatherDeserializer(const PrimitiveArray* arr) :
      arr_(arr) {}

  PyObject* Convert() {
    npy_intp dims[1] = {arr_->length};
    out_ = PyArray_SimpleNew(1, dims, feather_traits<TYPE>::npy_type);

    if (out_ == NULL) {
      // Error occurred, trust that SimpleNew set the error state
      return NULL;
    }
    ConvertValues();
    return out_;
  }

  void ConvertValues();

 private:
  const PrimitiveArray* arr_;
  PyObject* out_;
};

template <int TYPE>
inline void FeatherDeserializer<TYPE>::ConvertValues() {
  memcpy(PyArray_DATA(out_), arr_->values, arr_->length * ByteSize(arr_->type));
}

#define FROM_FEATHER_CASE(TYPE)                                 \
  case PrimitiveType::TYPE:                                     \
    {                                                           \
      FeatherDeserializer<PrimitiveType::TYPE> converter(&arr); \
      return converter.Convert();                               \
    }                                                           \
    break;

PyObject* primitive_to_pandas(const PrimitiveArray& arr) {
  switch(arr.type) {
    FROM_FEATHER_CASE(INT8);
    FROM_FEATHER_CASE(INT16);
    FROM_FEATHER_CASE(INT32);
    FROM_FEATHER_CASE(INT64);
    FROM_FEATHER_CASE(UINT8);
    FROM_FEATHER_CASE(UINT16);
    FROM_FEATHER_CASE(UINT32);
    FROM_FEATHER_CASE(UINT64);
    FROM_FEATHER_CASE(FLOAT);
    FROM_FEATHER_CASE(DOUBLE);
    // FROM_FEATHER_CASE(OBJECT);
    default:
      break;
  }
  PyErr_SetString(PyExc_NotImplementedError,
      "Feather type reading not implemented");
  return NULL;
}

} // namespace py

} // namespace feather
