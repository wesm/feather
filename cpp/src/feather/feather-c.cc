/*
 * Copyright 2016 Feather Developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>
#include <cstdint>
#include <string>

#include "feather/reader.h"
#include "feather/types.h"
#include "feather/writer.h"
#include "feather/feather-c.h"

using feather::Column;
using feather::ColumnType;
using feather::PrimitiveArray;
using feather::PrimitiveType;
using feather::Status;
using feather::TableReader;
using feather::TableWriter;

static PrimitiveType::type FromCFeatherType(feather_type ctype) {
  return static_cast<PrimitiveType::type>(static_cast<int>(ctype));
}

static feather_type ToCFeatherType(PrimitiveType::type type) {
  return static_cast<feather_type>(static_cast<int>(type));
}

static feather_column_type ToCFeatherColumnType(ColumnType::type type) {
  return static_cast<feather_column_type>(static_cast<int>(type));
}

static Status ToCFeatherArray(const PrimitiveArray& values, feather_array_t* out) {
  out->type = ToCFeatherType(values.type);
  out->length = values.length;
  out->null_count = values.null_count;

  out->nulls = values.nulls;
  out->values = values.values;
  out->offsets = values.offsets;

  return Status::OK();
}

static Status FromCFeatherArray(feather_array_t* carr, PrimitiveArray* out) {
  out->type = FromCFeatherType(carr->type);
  out->length = carr->length;
  out->null_count = carr->null_count;

  out->nulls = carr->nulls;
  out->values = carr->values;
  out->offsets = carr->offsets;

  return Status::OK();
}

#ifdef __cplusplus
extern "C" {
#if 0 /* confuse emacs indentation */
}
#endif
#endif

static feather_status get_feather_status(const Status& s) {
  if (s.ok()) {
    return FEATHER_OK;
  } else if (s.IsOutOfMemory()) {
    return FEATHER_OOM;
  } else if (s.IsKeyError()) {
    return FEATHER_KEY_ERROR;
  } else if (s.IsInvalid()) {
    return FEATHER_INVALID;
  } else if (s.IsIOError()) {
    return FEATHER_IO_ERROR;
  } else if (s.IsNotImplemented()) {
    return FEATHER_NOT_IMPLEMENTED;
  } else {
    return FEATHER_UNKNOWN;
  }
}

#define FEATHER_CHECK_STATUS(s) do {                \
    Status _s = (s);                                \
    if (!_s.ok()) return get_feather_status(_s);    \
  } while (0);

#define FEATHER_CHECK_MALLOC(ptr) do {          \
    if ((ptr) == nullptr) {                     \
      return FEATHER_OOM;                       \
    }                                           \
  } while (0);

/* Writer C API */

feather_status
feather_writer_open_file(const char* path, feather_writer_t** out) {
  std::unique_ptr<TableWriter> writer;
  try {
    std::string str_path(path);
    FEATHER_CHECK_STATUS(TableWriter::OpenFile(str_path, &writer));
  } catch (const std::exception& e) {
    (void) e;
    return FEATHER_OOM;
  }
  *out = reinterpret_cast<feather_writer_t*>(writer.release());
  return FEATHER_OK;
}

void feather_writer_set_num_rows(feather_writer_t* self, int64_t num_rows) {
  reinterpret_cast<TableWriter*>(self)->SetNumRows(num_rows);
}

feather_status
feather_writer_append_plain(feather_writer_t* self, const char* name,
    feather_array_t* values) {
  TableWriter* writer = reinterpret_cast<TableWriter*>(self);
  PrimitiveArray cpp_values;

  FEATHER_CHECK_STATUS(FromCFeatherArray(values, &cpp_values));

  try {
    std::string cpp_name(name);
    return get_feather_status(writer->AppendPlain(cpp_name, cpp_values));
  } catch (const std::exception& e) {
    (void) e;
    return FEATHER_OOM;
  }
}

feather_status
feather_writer_close(feather_writer_t* self) {
  return get_feather_status(
      reinterpret_cast<TableWriter*>(self)->Finalize());
}

feather_status
feather_writer_free(feather_writer_t* self) {
  delete reinterpret_cast<TableWriter*>(self);
  return FEATHER_OK;
}

/* Reader C API */

feather_status
feather_column_free(feather_column_t* self) {
  delete reinterpret_cast<Column*>(self->data);
  return FEATHER_OK;
}

feather_status
feather_reader_open_file(const char* path, feather_reader_t** out) {
  std::unique_ptr<TableReader> reader;
  try {
    std::string str_path(path);
    FEATHER_CHECK_STATUS(TableReader::OpenFile(str_path, &reader));
  } catch (const std::exception& e) {
    (void) e;
    return FEATHER_OOM;
  }
  *out = reinterpret_cast<feather_reader_t*>(reader.release());
  return FEATHER_OK;
}

int64_t
feather_reader_num_rows(feather_reader_t* self) {
  return reinterpret_cast<TableReader*>(self)->num_rows();
}

int64_t
feather_reader_num_columns(feather_reader_t* self) {
  return reinterpret_cast<TableReader*>(self)->num_columns();
}

feather_status
feather_reader_get_column(feather_reader_t* self, int i, feather_column_t* out) {
  TableReader* reader = reinterpret_cast<TableReader*>(self);
  std::unique_ptr<Column> col;
  FEATHER_CHECK_STATUS(reader->GetColumn(i, &col));

  out->type = ToCFeatherColumnType(col->type());
  out->name = col->name().c_str();

  FEATHER_CHECK_STATUS(ToCFeatherArray(col->values(), &out->values));

  out->data = reinterpret_cast<void*>(col.release());

  return FEATHER_OK;
}

feather_status
feather_reader_close(feather_reader_t* self) {
  return FEATHER_OK;
}

feather_status
feather_reader_free(feather_reader_t* self) {
  delete reinterpret_cast<TableReader*>(self);
  return FEATHER_OK;
}

#ifdef __cplusplus
#if 0 /* confuse emacs indentation */
{
#endif
}
#endif
