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

    cdef Status Status_OK "Status::OK"()

    cdef cppclass Status:
        Status()

        string ToString()

        c_bool ok()
        c_bool IsInvalid()
        c_bool IsOutOfMemory()
        c_bool IsIOError()

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
        ColumnType_CATEGORY" feather::ColumnType::CATEGORY"
        ColumnType_TIMESTAMP" feather::ColumnType::TIMESTAMP"

    enum Encoding" feather::Encoding::type":
        Encoding_PLAIN" feather::Encoding::PLAIN"
        Encoding_DICTIONARY" feather::Encoding::DICTIONARY"

    enum TimeUnit" feather::TimeUnit::type":
        TimeUnit_SECOND" feather::TimeUnit::SECOND"
        TimeUnit_MILLISECOND" feather::TimeUnit::MILLISECOND"
        TimeUnit_MICROSECOND" feather::TimeUnit::MICROSECOND"
        TimeUnit_NANOSECOND" feather::TimeUnit::NANOSECOND"

    cdef cppclass TimestampMetadata:
        TimeUnit unit
        string timezone

    cdef cppclass TimeMetadata:
        TimeUnit unit

    cdef cppclass Buffer:
        pass

    cdef cppclass PrimitiveArray:
        PrimitiveType type
        int64_t length
        int64_t null_count
        const uint8_t* nulls
        const uint8_t* values

        vector[shared_ptr[Buffer]] buffers

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

    cdef cppclass RandomAccessReader:
        int64_t Tell()
        void Seek(int64_t pos)

        shared_ptr[Buffer] ReadAt(int64_t position, int64_t nbytes)
        shared_ptr[Buffer] Read(int64_t nbytes)

        int64_t size()

    cdef cppclass LocalFileReader(RandomAccessReader):
        @staticmethod
        unique_ptr[LocalFileReader] Open(const string& path)

        void CloseFile()

        c_bool is_open()
        const string& path()

    cdef cppclass BufferReader(RandomAccessReader):
        BufferReader(const shared_ptr[Buffer]& buffer)



    cdef cppclass TableWriter:
        TableWriter()

        @staticmethod
        Status OpenFile(const string& abspath, unique_ptr[TableWriter]* out)

        void SetDescription(const string& desc)
        void SetNumRows(int64_t num_rows)

        Status AppendPlain(const string& name, const PrimitiveArray& values)
        Status AppendCategory(const string& name, const PrimitiveArray& values,
                              const PrimitiveArray& levels, c_bool ordered)

        Status AppendTimestamp(const string& name,
                               const PrimitiveArray& values,
                               const TimestampMetadata& meta)

        Status AppendDate(const string& name, const PrimitiveArray& values)

        Status AppendTime(const string& name, const PrimitiveArray& values,
                          const TimeMetadata& meta)

        Status Finalize()

    cdef cppclass CColumn" feather::Column":
        const PrimitiveArray& values()
        ColumnType type()
        string name()

    cdef cppclass CColumnMetadata" feather::metadata::Column":
        string name() const
        ColumnType type() const
        string user_metadata() const

    cdef cppclass CategoryColumn(CColumn):
        const PrimitiveArray& levels()

    cdef cppclass TimestampColumn(CColumn):
        TimeUnit unit()
        string timezone()

    cdef cppclass TableReader:
        TableReader(const shared_ptr[RandomAccessReader]& source)

        @staticmethod
        Status OpenFile(const string& abspath, unique_ptr[TableReader]* out)

        string GetDescription()
        c_bool HasDescription()

        int64_t num_rows()
        int64_t num_columns()

        Status GetColumn(int i, unique_ptr[CColumn]* out)
        Status GetColumnMetadata(int i, shared_ptr[CColumnMetadata]* out)
