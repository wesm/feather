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

#include <cmath>
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
  typedef uint8_t value_type;
  static constexpr PrimitiveType::type feather_type = PrimitiveType::BOOL;
  static constexpr bool supports_nulls = false;
  static inline bool isnull(uint8_t v) {
    return false;
  }
};

#define NPY_INT_DECL(TYPE, T)                                           \
  template <>                                                           \
  struct npy_traits<NPY_##TYPE> {                                       \
    typedef T value_type;                                               \
    static constexpr PrimitiveType::type feather_type = PrimitiveType::TYPE; \
    static constexpr bool supports_nulls = false;                       \
    static inline bool isnull(T v) {                                    \
      return false;                                                     \
    }                                                                   \
  };

NPY_INT_DECL(INT8, int8_t);
NPY_INT_DECL(INT16, int16_t);
NPY_INT_DECL(INT32, int32_t);
NPY_INT_DECL(INT64, int64_t);
NPY_INT_DECL(UINT8, uint8_t);
NPY_INT_DECL(UINT16, uint16_t);
NPY_INT_DECL(UINT32, uint32_t);
NPY_INT_DECL(UINT64, uint64_t);

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
  FeatherSerializer(PyArrayObject* arr, PyArrayObject* mask, PrimitiveArray* out) :
      arr_(arr),
      mask_(mask),
      out_(out) {}

  Status Convert();

  int stride() const {
    return PyArray_STRIDES(arr_)[0];
  }

  Status InitNullBitmap(uint8_t** bitmap) {
    int null_bytes = util::bytes_for_bits(out_->length);
    auto buffer = std::make_shared<OwnedMutableBuffer>();
    RETURN_NOT_OK(buffer->Resize(null_bytes));
    out_->buffers.push_back(buffer);
    memset(buffer->mutable_data(), 0, null_bytes);
    out_->nulls = buffer->data();

    *bitmap = buffer->mutable_data();
    return Status::OK();
  }

  bool is_strided() const {
    npy_intp* astrides = PyArray_STRIDES(arr_);
    return astrides[0] != PyArray_DESCR(arr_)->elsize;
  }

 private:
  Status ConvertData();

  PyArrayObject* arr_;
  PyArrayObject* mask_;
  PrimitiveArray* out_;
};

// Returns null count
static int64_t MaskToBitmap(PyArrayObject* mask, int64_t length, uint8_t* bitmap) {
  int64_t null_count = 0;
  const uint8_t* mask_values = static_cast<const uint8_t*>(PyArray_DATA(mask));
  // TODO(wesm): strided null mask
  for (int i = 0; i < length; ++i) {
    if (mask_values[i]) {
      ++null_count;
      util::set_bit(bitmap, i);
    }
  }
  return null_count;
}

template <int TYPE>
static int64_t ValuesToBitmap(const void* data, int64_t length, uint8_t* bitmap) {
  typedef npy_traits<TYPE> traits;
  typedef typename traits::value_type T;

  int64_t null_count = 0;
  const T* values = reinterpret_cast<const T*>(data);

  // TODO(wesm): striding
  for (int i = 0; i < length; ++i) {
    if (traits::isnull(values[i])) {
      ++null_count;
      util::set_bit(bitmap, i);
    }
  }

  return null_count;
}

template <int TYPE>
inline Status FeatherSerializer<TYPE>::Convert() {
  typedef npy_traits<TYPE> traits;

  out_->length = PyArray_SIZE(arr_);
  out_->type = traits::feather_type;

  uint8_t* bitmap = nullptr;
  if (mask_ != nullptr || traits::supports_nulls) {
    RETURN_NOT_OK(InitNullBitmap(&bitmap));
  }

  int64_t null_count = 0;
  if (mask_ != nullptr) {
    null_count = MaskToBitmap(mask_, out_->length, bitmap);
  } else if (traits::supports_nulls) {
    null_count = ValuesToBitmap<TYPE>(PyArray_DATA(arr_), out_->length, bitmap);
  }
  out_->null_count = null_count;

  RETURN_NOT_OK(ConvertData());
  return Status::OK();
}

template <int TYPE>
inline Status FeatherSerializer<TYPE>::ConvertData() {
  if (is_strided()) {
    return Status::Invalid("no support for strided data yet");
  }
  out_->values = static_cast<const uint8_t*>(PyArray_DATA(arr_));

  return Status::OK();
}

template <>
inline Status FeatherSerializer<NPY_BOOL>::ConvertData() {
  if (is_strided()) {
    return Status::Invalid("no support for strided data yet");
  }

  int nbytes = util::bytes_for_bits(out_->length);
  auto buffer = std::make_shared<OwnedMutableBuffer>();
  RETURN_NOT_OK(buffer->Resize(nbytes));
  out_->buffers.push_back(buffer);

  const uint8_t* values = reinterpret_cast<const uint8_t*>(PyArray_DATA(arr_));

  uint8_t* bitmap = buffer->mutable_data();

  memset(bitmap, 0, nbytes);
  for (int i = 0; i < out_->length; ++i) {
    if (values[i] > 0) {
      util::set_bit(bitmap, i);
    }
  }
  out_->values = bitmap;

  return Status::OK();
}

template <>
inline Status FeatherSerializer<NPY_OBJECT>::ConvertData() {
  return Status::Invalid("NYI");
}


#define TO_FEATHER_CASE(TYPE)                                   \
  case NPY_##TYPE:                                              \
    {                                                           \
      FeatherSerializer<NPY_##TYPE> converter(arr, mask, out);  \
      RETURN_NOT_OK(converter.Convert());                       \
    }                                                           \
    break;

Status pandas_masked_to_primitive(PyObject* ao, PyObject* mo,
    PrimitiveArray* out) {
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(ao);
  PyArrayObject* mask = nullptr;

  if (mo != nullptr) {
    mask = reinterpret_cast<PyArrayObject*>(mo);
  }

  if (arr->nd != 1) {
    return Status::Invalid("only handle 1-dimensional arrays");
  }

  switch(PyArray_DESCR(arr)->type_num) {
    TO_FEATHER_CASE(BOOL);
    TO_FEATHER_CASE(INT8);
    TO_FEATHER_CASE(INT16);
    TO_FEATHER_CASE(INT32);
    TO_FEATHER_CASE(INT64);
    TO_FEATHER_CASE(UINT8);
    TO_FEATHER_CASE(UINT16);
    TO_FEATHER_CASE(UINT32);
    TO_FEATHER_CASE(UINT64);
    TO_FEATHER_CASE(FLOAT32);
    TO_FEATHER_CASE(FLOAT64);
    // TO_FEATHER_CASE(OBJECT);
    default:
      std::stringstream ss;
      ss << "unsupported type " << PyArray_DESCR(arr)->type_num
         << std::endl;
      return Status::Invalid(ss.str());
  }
  return Status::OK();
}

Status pandas_to_primitive(PyObject* ao, PrimitiveArray* out) {
  return pandas_masked_to_primitive(ao, nullptr, out);
}

// ----------------------------------------------------------------------
// Deserialization

template <int TYPE>
struct feather_traits {
};

template <>
struct feather_traits<PrimitiveType::BOOL> {
  static constexpr int npy_type = NPY_BOOL;
  static constexpr bool supports_nulls = false;
  static constexpr bool is_boolean = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_floating = false;
};

#define INT_DECL(TYPE)                                      \
  template <>                                               \
  struct feather_traits<PrimitiveType::TYPE> {              \
    static constexpr int npy_type = NPY_##TYPE;             \
    static constexpr bool supports_nulls = false;           \
    static constexpr double na_value = NAN;                 \
    static constexpr bool is_boolean = false;               \
    static constexpr bool is_integer = true;                \
    static constexpr bool is_floating = false;              \
    typedef typename npy_traits<NPY_##TYPE>::value_type T;  \
  };

INT_DECL(INT8);
INT_DECL(INT16);
INT_DECL(INT32);
INT_DECL(INT64);
INT_DECL(UINT8);
INT_DECL(UINT16);
INT_DECL(UINT32);
INT_DECL(UINT64);

template <>
struct feather_traits<PrimitiveType::FLOAT> {
  static constexpr int npy_type = NPY_FLOAT32;
  static constexpr bool supports_nulls = true;
  static constexpr float na_value = NAN;
  static constexpr bool is_boolean = false;
  static constexpr bool is_integer = false;
  static constexpr bool is_floating = true;
  typedef typename npy_traits<NPY_FLOAT32>::value_type T;
};

template <>
struct feather_traits<PrimitiveType::DOUBLE> {
  static constexpr int npy_type = NPY_FLOAT64;
  static constexpr bool supports_nulls = true;
  static constexpr double na_value = NAN;
  static constexpr bool is_boolean = false;
  static constexpr bool is_integer = false;
  static constexpr bool is_floating = true;
  typedef typename npy_traits<NPY_FLOAT64>::value_type T;
};


template <int TYPE>
class FeatherDeserializer {
 public:
  FeatherDeserializer(const PrimitiveArray* arr) :
      arr_(arr) {}

  PyObject* Convert() {
    ConvertValues<TYPE>();
    return out_;
  }

  template <int T2>
  inline typename std::enable_if<feather_traits<T2>::is_floating, void>::type
  ConvertValues() {
    typedef typename feather_traits<T2>::T T;

    npy_intp dims[1] = {arr_->length};
    out_ = PyArray_SimpleNew(1, dims, feather_traits<T2>::npy_type);

    if (out_ == NULL) {
      // Error occurred, trust that SimpleNew set the error state
      return;
    }

    if (arr_->null_count > 0) {
      T* out_values = reinterpret_cast<T*>(PyArray_DATA(out_));
      const T* in_values = reinterpret_cast<const T*>(arr_->values);
      for (int64_t i = 0; i < arr_->length; ++i) {
        out_values[i] = util::get_bit(arr_->nulls, i) ? NAN : in_values[i];
      }
    } else {
      memcpy(PyArray_DATA(out_), arr_->values,
          arr_->length * ByteSize(arr_->type));
    }
  }

  // Integer specialization
  template <int T2>
  inline typename std::enable_if<feather_traits<T2>::is_integer, void>::type
  ConvertValues() {
    typedef typename feather_traits<T2>::T T;

    npy_intp dims[1] = {arr_->length};
    if (arr_->null_count > 0) {
      out_ = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
      if (out_ == NULL)  return;

      // Upcast to double, set NaN as appropriate
      double* out_values = reinterpret_cast<double*>(PyArray_DATA(out_));
      const T* in_values = reinterpret_cast<const T*>(arr_->values);
      for (int i = 0; i < arr_->length; ++i) {
        out_values[i] = util::get_bit(arr_->nulls, i) ? NAN : in_values[i];
      }
    } else {
      out_ = PyArray_SimpleNew(1, dims, feather_traits<TYPE>::npy_type);
      if (out_ == NULL)  return;
      memcpy(PyArray_DATA(out_), arr_->values, arr_->length * ByteSize(arr_->type));
    }
  }

  // Boolean specialization
  template <int T2>
  inline typename std::enable_if<feather_traits<T2>::is_boolean, void>::type
  ConvertValues() {
    npy_intp dims[1] = {arr_->length};
    if (arr_->null_count > 0) {
      out_ = PyArray_SimpleNew(1, dims, NPY_OBJECT);
      if (out_ == NULL)  return;
      PyObject** out_values = reinterpret_cast<PyObject**>(PyArray_DATA(out_));
      for (int64_t i = 0; i < arr_->length; ++i) {
        if (util::get_bit(arr_->nulls, i)) {
          Py_INCREF(Py_None);
          out_values[i] = Py_None;
        } else if (util::get_bit(arr_->values, i)) {
          // True
          Py_INCREF(Py_True);
          out_values[i] = Py_True;
        } else {
          // False
          Py_INCREF(Py_False);
          out_values[i] = Py_False;
        }
      }
    } else {
      out_ = PyArray_SimpleNew(1, dims, feather_traits<TYPE>::npy_type);
      if (out_ == NULL)  return;

      uint8_t* out_values = reinterpret_cast<uint8_t*>(PyArray_DATA(out_));
      for (int64_t i = 0; i < arr_->length; ++i) {
        out_values[i] = util::get_bit(arr_->values, i) ? 1 : 0;
      }
    }
  }

 private:
  const PrimitiveArray* arr_;
  PyObject* out_;
};


#define FROM_FEATHER_CASE(TYPE)                                 \
  case PrimitiveType::TYPE:                                     \
    {                                                           \
      FeatherDeserializer<PrimitiveType::TYPE> converter(&arr); \
      return converter.Convert();                               \
    }                                                           \
    break;

PyObject* primitive_to_pandas(const PrimitiveArray& arr) {
  switch(arr.type) {
    FROM_FEATHER_CASE(BOOL);
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
    // FROM_FEATHER_CASE(UTF8);
    // FROM_FEATHER_CASE(CATEGORY);
    default:
      break;
  }
  PyErr_SetString(PyExc_NotImplementedError,
      "Feather type reading not implemented");
  return NULL;
}

} // namespace py

} // namespace feather
