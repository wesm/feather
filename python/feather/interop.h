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

#define NPY_INT_DECL(TYPE, FTYPE, T)                                    \
  template <>                                                           \
  struct npy_traits<NPY_##TYPE> {                                       \
    typedef T value_type;                                               \
    static constexpr PrimitiveType::type feather_type = PrimitiveType::FTYPE; \
    static constexpr bool supports_nulls = false;                       \
    static inline bool isnull(T v) {                                    \
      return false;                                                     \
    }                                                                   \
  };

NPY_INT_DECL(INT8, INT8, int8_t);
NPY_INT_DECL(INT16, INT16, int16_t);
NPY_INT_DECL(INT32, INT32, int32_t);
NPY_INT_DECL(INT64, INT64, int64_t);

NPY_INT_DECL(UINT8, UINT8, uint8_t);
NPY_INT_DECL(UINT16, UINT16, uint16_t);
NPY_INT_DECL(UINT32, UINT32, uint32_t);
NPY_INT_DECL(UINT64, UINT64, uint64_t);

#if NPY_INT64 != NPY_LONGLONG
NPY_INT_DECL(LONGLONG, INT64, int64_t);
NPY_INT_DECL(ULONGLONG, UINT64, uint64_t);
#endif

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
      out_(out),
      null_bitmap_(nullptr) {}

  Status Convert();

  int stride() const {
    return PyArray_STRIDES(arr_)[0];
  }

  Status InitNullBitmap() {
    int null_bytes = util::bytes_for_bits(out_->length);
    auto buffer = std::make_shared<OwnedMutableBuffer>();
    RETURN_NOT_OK(buffer->Resize(null_bytes));
    out_->buffers.push_back(buffer);
    util::fill_buffer(buffer->mutable_data(), 0, null_bytes);
    out_->nulls = buffer->data();

    null_bitmap_ = buffer->mutable_data();
    return Status::OK();
  }

  bool is_strided() const {
    npy_intp* astrides = PyArray_STRIDES(arr_);
    return astrides[0] != PyArray_DESCR(arr_)->elsize;
  }

 private:
  Status ConvertData();

  Status ConvertObjectStrings() {
    PyObject** objects = reinterpret_cast<PyObject**>(PyArray_DATA(arr_));

    auto offsets_buffer = std::make_shared<OwnedMutableBuffer>();
    RETURN_NOT_OK(offsets_buffer->Resize(sizeof(int32_t) * (out_->length + 1)));
    int32_t* offsets = reinterpret_cast<int32_t*>(offsets_buffer->mutable_data());

    BufferBuilder data_builder;

    Status s;
    PyObject* obj;
    int length;
    int offset = 0;
    int64_t null_count = 0;
    for (int64_t i = 0; i < out_->length; ++i) {
      obj = objects[i];
      if (PyUnicode_Check(obj)) {
        obj = PyUnicode_AsUTF8String(obj);
        if (obj == NULL) {
          PyErr_Clear();
          return Status::Invalid("failed converting unicode to UTF8");
        }
        length = PyBytes_GET_SIZE(obj);
        s = data_builder.Append(
            reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(obj)), length);
        Py_DECREF(obj);
        if (!s.ok()) {
          return s;
        }
        util::set_bit(null_bitmap_, i);
      } else if (PyBytes_Check(obj)) {
        length = PyBytes_GET_SIZE(obj);
        RETURN_NOT_OK(data_builder.Append(
                reinterpret_cast<const uint8_t*>(PyBytes_AS_STRING(obj)), length));
        util::set_bit(null_bitmap_, i);
      } else {
        // NULL
        // No change to offset
        length = 0;
        ++null_count;
      }
      offsets[i] = offset;
      offset += length;
    }
    // End offset
    offsets[out_->length] = offset;

    std::shared_ptr<Buffer> data_buffer = data_builder.Finish();
    out_->type = PrimitiveType::UTF8;
    out_->values = data_buffer->data();
    out_->buffers.push_back(data_buffer);

    out_->offsets = offsets;
    out_->buffers.push_back(offsets_buffer);
    out_->null_count = null_count;

    return Status::OK();
  }

  Status ConvertBooleans() {
    out_->type = PrimitiveType::BOOL;
    PyObject** objects = reinterpret_cast<PyObject**>(PyArray_DATA(arr_));

    int nbytes = util::bytes_for_bits(out_->length);
    auto buffer = std::make_shared<OwnedMutableBuffer>();
    RETURN_NOT_OK(buffer->Resize(nbytes));
    out_->buffers.push_back(buffer);
    uint8_t* bitmap = buffer->mutable_data();
    util::fill_buffer(bitmap, 0, nbytes);

    int64_t null_count = 0;
    for (int64_t i = 0; i < out_->length; ++i) {
      if (objects[i] == Py_True) {
        util::set_bit(bitmap, i);
        util::set_bit(null_bitmap_, i);
      } else if (objects[i] == Py_False) {
        util::set_bit(null_bitmap_, i);
      } else {
        ++null_count;
      }
    }
    out_->type = PrimitiveType::BOOL;
    out_->values = bitmap;
    out_->null_count = null_count;

    return Status::OK();
  }

  PyArrayObject* arr_;
  PyArrayObject* mask_;
  PrimitiveArray* out_;

  uint8_t* null_bitmap_;
};

// Returns null count
static int64_t MaskToBitmap(PyArrayObject* mask, int64_t length, uint8_t* bitmap) {
  int64_t null_count = 0;
  const uint8_t* mask_values = static_cast<const uint8_t*>(PyArray_DATA(mask));
  // TODO(wesm): strided null mask
  for (int i = 0; i < length; ++i) {
    if (mask_values[i]) {
      ++null_count;
    } else {
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
    } else {
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

  if (mask_ != nullptr || traits::supports_nulls) {
    RETURN_NOT_OK(InitNullBitmap());
  }

  int64_t null_count = 0;
  if (mask_ != nullptr) {
    null_count = MaskToBitmap(mask_, out_->length, null_bitmap_);
  } else if (traits::supports_nulls) {
    null_count = ValuesToBitmap<TYPE>(PyArray_DATA(arr_), out_->length, null_bitmap_);
  }
  out_->null_count = null_count;

  RETURN_NOT_OK(ConvertData());
  return Status::OK();
}

PyObject* numpy_nan = nullptr;

void set_numpy_nan(PyObject* nan) {
  Py_INCREF(nan);
  numpy_nan = nan;
}

static inline bool PyObject_is_null(const PyObject* obj) {
  if (obj == Py_None || obj == numpy_nan) {
    return true;
  }
  if (PyFloat_Check(obj)) {
    double val = PyFloat_AS_DOUBLE(obj);
    return val != val;
  }
  return false;
}

static inline bool PyObject_is_string(const PyObject* obj) {
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_Check(obj) || PyBytes_Check(obj);
#else
  return PyString_Check(obj) || PyUnicode_Check(obj);
#endif
}

template <>
inline Status FeatherSerializer<NPY_OBJECT>::Convert() {
  // Python object arrays are annoying, since we could have one of:
  //
  // * Strings
  // * Booleans with nulls
  // * Mixed type (not supported at the moment by feather format)
  //
  // Additionally, nulls may be encoded either as np.nan or None. So we have to
  // do some type inference and conversion

  out_->length = PyArray_SIZE(arr_);
  RETURN_NOT_OK(InitNullBitmap());

  // TODO: mask not supported here
  const PyObject** objects = reinterpret_cast<const PyObject**>(PyArray_DATA(arr_));

  for (int64_t i = 0; i < out_->length; ++i) {
    if (PyObject_is_null(objects[i])) {
      continue;
    } else if (PyObject_is_string(objects[i])) {
      return ConvertObjectStrings();
    } else if (PyBool_Check(objects[i])) {
      return ConvertBooleans();
    } else {
      std::stringstream ss;
      ss << "unhandled python type, index " << i;
      return Status::Invalid(ss.str());
    }
  }

  return Status::Invalid("Unable to infer type of object array, were all null");
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

  util::fill_buffer(bitmap, 0, nbytes);
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

  int type_num = PyArray_DESCR(arr)->type_num;

#if (NPY_INT64 == NPY_LONGLONG) && (NPY_SIZEOF_LONGLONG == 8)
  // GH #129, on i386 / Apple Python, both LONGLONG and INT64 can be observed
  // in the wild, which is buggy. We set U/LONGLONG to U/INT64 so things work
  // properly.
  if (type_num == NPY_LONGLONG) {
    type_num = NPY_INT64;
  }
  if (type_num == NPY_ULONGLONG) {
    type_num = NPY_UINT64;
  }
#endif

  switch(type_num) {
    TO_FEATHER_CASE(BOOL);
    TO_FEATHER_CASE(INT8);
    TO_FEATHER_CASE(INT16);
    TO_FEATHER_CASE(INT32);
    TO_FEATHER_CASE(INT64);
#if (NPY_INT64 != NPY_LONGLONG)
    TO_FEATHER_CASE(LONGLONG);
#endif
    TO_FEATHER_CASE(UINT8);
    TO_FEATHER_CASE(UINT16);
    TO_FEATHER_CASE(UINT32);
    TO_FEATHER_CASE(UINT64);
#if (NPY_UINT64 != NPY_ULONGLONG)
    TO_FEATHER_CASE(ULONGLONG);
#endif
    TO_FEATHER_CASE(FLOAT32);
    TO_FEATHER_CASE(FLOAT64);
    TO_FEATHER_CASE(OBJECT);
    default:
      std::stringstream ss;
      ss << "unsupported type " << type_num
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

template <>
struct feather_traits<PrimitiveType::UTF8> {
  static constexpr int npy_type = NPY_OBJECT;
  static constexpr bool supports_nulls = true;
  static constexpr bool is_boolean = false;
  static constexpr bool is_integer = false;
  static constexpr bool is_floating = false;
};


static inline PyObject* make_pystring(const uint8_t* data, int32_t length) {
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromStringAndSize(reinterpret_cast<const char*>(data), length);
#else
  return PyString_FromStringAndSize(reinterpret_cast<const char*>(data), length);
#endif
}

template <int TYPE>
class FeatherDeserializer {
 public:
  FeatherDeserializer(const PrimitiveArray* arr) :
      arr_(arr) {}

  PyObject* Convert() {
    ConvertValues<TYPE>();
    return out_;
  }

  // Floating point specialization
  template <int T2>
  inline typename std::enable_if<feather_traits<T2>::is_floating, void>::type
  ConvertValues() {
    typedef typename feather_traits<T2>::T T;

    npy_intp dims[1] = {static_cast<npy_intp>(arr_->length)};
    out_ = PyArray_SimpleNew(1, dims, feather_traits<T2>::npy_type);

    if (out_ == NULL) {
      // Error occurred, trust that SimpleNew set the error state
      return;
    }

    if (arr_->null_count > 0) {
      T* out_values = reinterpret_cast<T*>(PyArray_DATA(out_));
      const T* in_values = reinterpret_cast<const T*>(arr_->values);
      for (int64_t i = 0; i < arr_->length; ++i) {
        out_values[i] = util::bit_not_set(arr_->nulls, i) ? NAN : in_values[i];
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

    npy_intp dims[1] = {static_cast<npy_intp>(arr_->length)};
    if (arr_->null_count > 0) {
      out_ = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
      if (out_ == NULL)  return;

      // Upcast to double, set NaN as appropriate
      double* out_values = reinterpret_cast<double*>(PyArray_DATA(out_));
      const T* in_values = reinterpret_cast<const T*>(arr_->values);
      for (int i = 0; i < arr_->length; ++i) {
        out_values[i] = util::bit_not_set(arr_->nulls, i) ? (double) NAN : in_values[i];
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
    npy_intp dims[1] = {static_cast<npy_intp>(arr_->length)};
    if (arr_->null_count > 0) {
      out_ = PyArray_SimpleNew(1, dims, NPY_OBJECT);
      if (out_ == NULL)  return;
      PyObject** out_values = reinterpret_cast<PyObject**>(PyArray_DATA(out_));
      for (int64_t i = 0; i < arr_->length; ++i) {
        if (util::bit_not_set(arr_->nulls, i)) {
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

  // UTF8
  template <int T2>
  inline typename std::enable_if<T2 == PrimitiveType::UTF8, void>::type
  ConvertValues() {
    npy_intp dims[1] = {static_cast<npy_intp>(arr_->length)};
    out_ = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    if (out_ == NULL)  return;
    PyObject** out_values = reinterpret_cast<PyObject**>(PyArray_DATA(out_));
    int32_t offset;
    int32_t length;
    if (arr_->null_count > 0) {
      for (int64_t i = 0; i < arr_->length; ++i) {
        if (util::bit_not_set(arr_->nulls, i)) {
          Py_INCREF(Py_None);
          out_values[i] = Py_None;
        } else {
          offset = arr_->offsets[i];
          length = arr_->offsets[i + 1] - offset;
          out_values[i] = make_pystring(arr_->values + offset, length);
          if (out_values[i] == nullptr) return;
        }
      }
    } else {
      for (int64_t i = 0; i < arr_->length; ++i) {
        offset = arr_->offsets[i];
        length = arr_->offsets[i + 1] - offset;
        out_values[i] = make_pystring(arr_->values + offset, length);
        if (out_values[i] == nullptr) return;
      }
    }
  }
 private:
  const PrimitiveArray* arr_;
  PyObject* out_;
};

PyObject* get_null_mask(const PrimitiveArray& arr) {
  npy_intp dims[1] = {static_cast<npy_intp>(arr.length)};
  PyObject* out = PyArray_SimpleNew(1, dims, NPY_BOOL);
  if (out == NULL) return out;

  uint8_t* out_values = reinterpret_cast<uint8_t*>(PyArray_DATA(out));
  if (arr.null_count > 0) {
    for (int64_t i = 0; i < arr.length; ++i) {
      out_values[i] = util::bit_not_set(arr.nulls, i);
    }
  } else {
    for (int64_t i = 0; i < arr.length; ++i) {
      out_values[i] = 0;
    }
  }
  return out;
}

#define FROM_RAW_FEATHER_CASE(TYPE)                             \
  case PrimitiveType::TYPE:                                     \
    {                                                           \
      npy_intp dims[1] = {static_cast<npy_intp>(arr.length)};   \
      PyObject* out = PyArray_SimpleNew(1, dims,                \
        feather_traits<PrimitiveType::TYPE>::npy_type);         \
      if (out == NULL)  return out;                             \
      memcpy(PyArray_DATA(out), arr.values,                     \
        arr.length * ByteSize(arr.type));                       \
      return out;                                               \
    }                                                           \
    break;

PyObject* raw_primitive_to_pandas(const PrimitiveArray& arr) {
  switch(arr.type) {
    FROM_RAW_FEATHER_CASE(INT8);
    FROM_RAW_FEATHER_CASE(INT16);
    FROM_RAW_FEATHER_CASE(INT32);
    FROM_RAW_FEATHER_CASE(INT64);
    FROM_RAW_FEATHER_CASE(UINT8);
    FROM_RAW_FEATHER_CASE(UINT16);
    FROM_RAW_FEATHER_CASE(UINT32);
    FROM_RAW_FEATHER_CASE(UINT64);
    default:
      break;
  }
  PyErr_SetString(PyExc_NotImplementedError,
      "Feather type raw reading not implemented");
  return NULL;
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
    FROM_FEATHER_CASE(UTF8);
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
