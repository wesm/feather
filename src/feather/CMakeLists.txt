# Copyright 2012 Feather Developers
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

install(FILES
  buffer.h
  common.h
  exception.h
  io.h
  metadata.h
  reader.h
  types.h
  writer.h
  DESTINATION include/feather)

add_library(feather_test_main
  test_main.cc)

if (APPLE)
  target_link_libraries(feather_test_main
    gtest
    dl)
  set_target_properties(feather_test_main
        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  target_link_libraries(feather_test_main
    dl
    gtest
    pthread
  )
endif()

ADD_FEATHER_TEST(io-test)
ADD_FEATHER_TEST(metadata-test)
ADD_FEATHER_TEST(writer-test)
