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

#include "feather/reader.h"
#include "feather/types.h"
#include "feather/writer.h"
#include "feather/feather-c.h"

using namespace feather;

#ifdef __cplusplus
extern "C" {
#if 0 /* confuse emacs indentation */
}
#endif
#endif

static feather_status get_feather_status (const Status& s) {

}

#define FEATHER_CHECK_STATUS(s) do {                \
    Status _s = (s);                                \
    if (!_s.ok()) return get_feather_status(_s);    \
  } while (0);

#define FEATHER_CHECK_STATUS(s)

feather_status
feather_reader_open_file(const char* path, feather_reader_t** out) {
  try {
    std::string str_path(path);
    std::unique_ptr<TableReader> reader;

    TableReader::OpenFile(path, &reader);

    *out = reinterpret_cast<feather_reader_t*>(
  } catch (const std::exception& e) {
    return FEATHER_OOM;
  }
}

#ifdef __cplusplus
#if 0 /* confuse emacs indentation */
{
#endif
}
#endif
