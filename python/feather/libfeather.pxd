# Copyright 2016 Feather Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++

from libc.stdint cimport *
from libcpp cimport bool as c_bool
from libcpp.string cimport string
from libcpp.vector cimport vector

# This must be included for cerr and other things to work
cdef extern from "<iostream>":
    pass

cdef extern from "<memory>" namespace "std" nogil:

    cdef cppclass shared_ptr[T]:
        T* get()
        void reset()
        void reset(T* p)

    shared_ptr[T] make_shared[T](T* p)

    cdef cppclass unique_ptr[T]:
        T* get()
        void reset()
        void reset(T* p)

cdef extern from "feather/api.h" namespace "feather" nogil:

    enum PrimitiveType" feather::PrimitiveType::type":
        PrimitiveType_BOOL" feather::PrimitiveType::BOOL"
        PrimitiveType_INT8" feather::PrimitiveType::INT8"
        PrimitiveType_INT16" feather::PrimitiveType::INT16"
        PrimitiveType_INT32" feather::PrimitiveType::INT32"
        PrimitiveType_INT64" feather::PrimitiveType::INT64"
        PrimitiveType_UINT8" feather::PrimitiveType::UINT8"
        PrimitiveType_UINT16" feather::PrimitiveType::UINT16"
        PrimitiveType_UINT32" feather::PrimitiveType::UINT32"
        PrimitiveType_UINT64" feather::PrimitiveType::UINT64"
        PrimitiveType_FLOAT" feather::PrimitiveType::FLOAT"
        PrimitiveType_DOUBLE" feather::PrimitiveType::DOUBLE"
        PrimitiveType_UTF8" feather::PrimitiveType::UTF8"
        PrimitiveType_BINARY" feather::PrimitiveType::BINARY"
        PrimitiveType_CATEGORY" feather::PrimitiveType::CATEGORY"

    enum ColumnType" feather::ColumnType::type":
        ColumnType_PRIMITIVE" feather::ColumnType::PRIMITIVE"

    enum Encoding" feather::Encoding::type":
        Encoding_PLAIN" feather::Encoding::PLAIN"
        Encoding_DICTIONARY" feather::Encoding::DICTIONARY"

    cdef cppclass Buffer:
        pass

    cdef cppclass PrimitiveArray:
        PrimitiveType type
        int64_t length
        int64_t null_count
        const uint8_t* nulls
        const uint8_t* values

        # For UTF8 and BINARY, not used otherwise
        const int32_t* offsets
        c_bool Equals(const PrimitiveArray& other)

    cdef cppclass CategoryArray:
        PrimitiveArray indices
        PrimitiveArray levels
        c_bool ordered

    cdef cppclass DictEncodedArray:
        PrimitiveArray dict_values
        PrimitiveArray indices

    cdef cppclass InputStream:
        pass

    cdef cppclass OutputStream:
        void Close()
        int64_t Tell()
        void Write(const uint8_t* data, int64_t length)

    cdef cppclass InMemoryOutputStream(OutputStream):
        InMemoryOutputStream(int64_t initial_capacity)
        shared_ptr[Buffer] Finish()

    cdef cppclass TableWriter:
        TableWriter(const shared_ptr[OutputStream]& stream)

        void SetDescription(const string& desc)
        void SetNumRows(int64_t num_rows)

        void AppendPlain(const string& name, const PrimitiveArray& values)
        void Finalize()
