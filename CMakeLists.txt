cmake_minimum_required(VERSION 3.28)
project(qoraholo VERSION 0.1.0 LANGUAGES CXX)

option(QORAHOLO_CUDA "Build the CUDA backend" OFF)
option(QORAHOLO_BUILD_TESTS "Build the GoogleTest suite" ${PROJECT_IS_TOP_LEVEL})
option(QORAHOLO_BUILD_BENCH "Build the benchmark" ${PROJECT_IS_TOP_LEVEL})
option(QORAHOLO_BUILD_EXAMPLES "Build the C++ examples" ${PROJECT_IS_TOP_LEVEL})
option(QORAHOLO_BUILD_PYTHON "Build the nanobind module" OFF)
option(QORAHOLO_WERROR "Treat warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(QORAHOLO_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
endif()

include(FetchContent)
FetchContent_Declare(
  pocketfft
  URL https://github.com/mreineck/pocketfft/archive/4d865cba24359b05b37da4e75aee24498c3b78e7.tar.gz
  URL_HASH SHA256=db236e44b1020dd0b415e5040ce10d8be604fefb7ce83ec0b32bc0f78975aa2f
)
FetchContent_MakeAvailable(pocketfft)

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)

add_library(qoraholo_warnings INTERFACE)
if(MSVC)
  target_compile_options(qoraholo_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /utf-8 $<$<BOOL:${QORAHOLO_WERROR}>:/WX>>)
else()
  target_compile_options(qoraholo_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow
      $<$<BOOL:${QORAHOLO_WERROR}>:-Werror>>)
endif()
target_compile_options(qoraholo_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings>)

add_library(qoraholo
  src/api.cpp
  src/phantom.cpp
  src/pipeline.cpp
  src/cpu/fft.cpp
  src/cpu/field.cpp
  src/cpu/odt.cpp
  src/cpu/tv.cpp
  src/cpu/unwrap.cpp
)
add_library(qoraholo::qoraholo ALIAS qoraholo)
target_include_directories(qoraholo
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(qoraholo SYSTEM PRIVATE ${pocketfft_SOURCE_DIR})
target_link_libraries(qoraholo PRIVATE qoraholo_warnings Threads::Threads)
if(NOT MSVC)
  target_compile_options(qoraholo PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wconversion -Wno-sign-conversion>)
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(qoraholo PRIVATE OpenMP::OpenMP_CXX)
endif()
if(QORAHOLO_CUDA)
  target_sources(qoraholo PRIVATE
    src/cuda/runtime.cu
    src/cuda/field.cu
    src/cuda/unwrap.cu
    src/cuda/odt.cu
    src/cuda/tv.cu
    src/cuda/pipeline.cu
  )
  target_compile_definitions(qoraholo PUBLIC QORAHOLO_HAS_CUDA)
  target_link_libraries(qoraholo PRIVATE CUDA::cudart CUDA::cufft)
  set_target_properties(qoraholo PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(QORAHOLO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(QORAHOLO_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(QORAHOLO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(QORAHOLO_BUILD_PYTHON)
  add_subdirectory(python)
endif()
