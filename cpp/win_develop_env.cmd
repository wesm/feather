@ECHO OFF

set SOURCE_DIR=%~dp0
call "%SOURCE_DIR%\thirdparty\win_versions.cmd"
set FLATBUFFERS_HOME=%SOURCE_DIR%thirdparty\installed
set GTEST_HOME=%SOURCE_DIR%thirdparty\installed
