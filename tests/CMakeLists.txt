FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(qoraholo_tests
  test_dct.cpp
  test_field.cpp
  test_unwrap.cpp
  test_odt.cpp
  test_cuda.cpp
  test_pipeline.cpp
  test_bpm.cpp
)
target_include_directories(qoraholo_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(qoraholo_tests SYSTEM PRIVATE ${pocketfft_SOURCE_DIR})
target_link_libraries(qoraholo_tests PRIVATE qoraholo qoraholo_warnings GTest::gtest_main Threads::Threads)

include(GoogleTest)
gtest_discover_tests(qoraholo_tests DISCOVERY_TIMEOUT 60)
