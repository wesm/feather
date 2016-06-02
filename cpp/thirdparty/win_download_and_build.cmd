@ECHO OFF

set start=%CD%
set TP_DIR=%~dp0
set PREFIX=%TP_DIR%\installed

call "%TP_DIR%\win_versions.cmd"

mkdir %PREFIX%
mkdir %PREFIX%\include
mkdir %PREFIX%\lib
mkdir %PREFIX%\bin

cd %TP_DIR%
git clone https://github.com/google/googletest.git --depth 1 --branch "release-%GTEST_VERSION%" %GTEST_BASEDIR%
cd %GTEST_BASEDIR%
mkdir build-release
cd build-release
cmake -G "Visual Studio 14 Win64" -DCMAKE_INSTALL_PREFIX=%PREFIX% -D gtest_force_shared_crt=ON ..
cmake --build . --config Release
copy Release\gtest.lib %PREFIX%\lib /Y
cd ..
xcopy include %PREFIX%\include /E /Y


cd %TP_DIR%
git clone https://github.com/google/flatbuffers.git --depth 1 --branch "v%FLATBUFFERS_VERSION%" %FLATBUFFERS_BASEDIR%
cd %FLATBUFFERS_BASEDIR%
mkdir build-release
cd build-release
cmake -G "Visual Studio 14 Win64" -DCMAKE_INSTALL_PREFIX=%PREFIX% -DFLATBUFFERS_BUILD_TESTS=OFF ..
cmake --build . --config Release
cmake --build . --config Release --target Install

cd %start%

