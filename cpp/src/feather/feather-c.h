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

#ifndef FEATHER_FEATHER_C_H
#define FEATHER_FEATHER_C_H

#ifdef __cplusplus
extern "C" {
#if 0 /* confuse emacs indentation */
}
#endif
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum {
  FEATHER_OK = 0,
  FEATHER_OOM = 1,
  FEATHER_KEY_ERROR = 2,
  FEATHER_INVALID = 3,
  FEATHER_IO_ERROR = 4,
  FEATHER_NOT_IMPLEMENTED = 10,
  FEATHER_UNKNOWN = 50
} feather_status;

typedef enum {
  FEATHER_BOOL = 0,
  FEATHER_INT8 = 1,
  FEATHER_INT16 = 2,
  FEATHER_INT32 = 3,
  FEATHER_INT64 = 4,
  FEATHER_UINT8 = 5,
  FEATHER_UINT16 = 6,
  FEATHER_UINT32 = 7,
  FEATHER_UINT64 = 8,
  FEATHER_FLOAT = 9,
  FEATHER_DOUBLE = 10,
  FEATHER_UTF8 = 11,
  FEATHER_BINARY = 12
} feather_type;

/*
  Column C API
 */

typedef enum {
  FEATHER_COLUMN_PRIMITIVE = 0,
  FEATHER_COLUMN_CATEGORY = 1,
  FEATHER_COLUMN_TIMESTAMP = 2,
  FEATHER_COLUMN_DATE = 3,
  FEATHER_COLUMN_TIME = 4,
} feather_column_type;

typedef enum {
  FEATHER_UNIT_SECOND = 0,
  FEATHER_UNIT_MILLISECOND = 1,
  FEATHER_UNIT_MICROSECOND = 2,
  FEATHER_UNIT_NANOSECOND = 3
} feather_time_unit;

typedef struct {
  feather_type type;
  int64_t length;
  int64_t null_count;

  const uint8_t* nulls;
  const uint8_t* values;

  const int32_t* offsets;
} feather_array_t;

typedef struct {
  feather_array_t indices;
  feather_array_t levels;
  int ordered;
} feather_category_t;

typedef struct {
  feather_array_t levels;
  int ordered;
} feather_category_data_t;

typedef struct {
  const char* timezone;
  feather_time_unit unit;
} feather_timestamp_data_t;

typedef struct {
  feather_time_unit unit;
} feather_time_data_t;

typedef struct {
  feather_column_type type;
  const char* name;
  feather_array_t values;

  void* data;
  void* type_metadata;
} feather_column_t;

feather_status
feather_column_free(feather_column_t* self);

/*
 *************************************************
  TableWriter C API
 *************************************************
*/

typedef void feather_writer_t;

feather_status
feather_writer_open_file(const char* path, feather_writer_t** out);

void
feather_writer_set_num_rows(feather_writer_t* self, int64_t num_rows);

/* Write primitive array */
feather_status
feather_writer_append_plain(feather_writer_t* self, const char* name,
    feather_array_t* values);

feather_status
feather_writer_append_category(feather_writer_t* self, const char* name,
    feather_array_t* values, feather_array_t* levels, int ordered);

feather_status
feather_writer_append_timestamp(feather_writer_t* self, const char* name,
    feather_array_t* values, const char* timezone,
    feather_time_unit unit);

feather_status
feather_writer_append_time(feather_writer_t* self, const char* name,
    feather_array_t* values, feather_time_unit unit);

feather_status
feather_writer_append_date(feather_writer_t* self, const char* name,
    feather_array_t* values);

/* Write file metadata and footer */
feather_status
feather_writer_close(feather_writer_t* self);

/* Close file if any, and deallocate TableWriter */
feather_status
feather_writer_free(feather_writer_t* self);

/*
 *************************************************
  TableReader C API
 *************************************************
*/

typedef void feather_reader_t;

feather_status
feather_reader_open_file(const char* path, feather_reader_t** out);

int64_t
feather_reader_num_rows(feather_reader_t* self);

int64_t
feather_reader_num_columns(feather_reader_t* self);

/*
 * Retrieve the column metadata and data pointers from the file. Call
 * feather_column_free when finished with the column.
 */
feather_status
feather_reader_get_column(feather_reader_t* self, int i, feather_column_t* out);

feather_status
feather_reader_close(feather_reader_t* self);

feather_status
feather_reader_free(feather_reader_t* self);

#ifdef __cplusplus
#if 0 /* confuse emacs indentation */
{
#endif
}
#endif

#endif /* FEATHER_FEATHER_C_H */
