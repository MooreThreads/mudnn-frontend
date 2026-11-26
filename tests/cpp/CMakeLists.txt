cmake_minimum_required(VERSION 3.12)

project(mudnn_frontend_framework_test LANGUAGES CXX)

set(CMAKE_MODULE_PATH /usr/local/musa/cmake ${CMAKE_MODULE_PATH})
find_package(MUSA REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

function(mudnn_frontend_configure_test_target target_name)
  target_compile_features(${target_name} PRIVATE cxx_std_17)
  target_compile_options(${target_name} PRIVATE
    -Wall
    -Wextra
    -Werror
    -Wno-unused-variable
    -Wno-sequence-point
  )

  if(EXISTS "${CMAKE_BINARY_DIR}/export_headers/c")
    target_include_directories(${target_name} BEFORE
      PRIVATE ${CMAKE_BINARY_DIR}/export_headers/c
    )
  endif()

  target_include_directories(${target_name} BEFORE
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  )

  if(EXISTS "${MUDNN_BACKEND_INCLUDE}")
    target_include_directories(${target_name}
      PRIVATE ${MUDNN_BACKEND_INCLUDE}
    )
  endif()

  target_include_directories(${target_name} PRIVATE
    ${MUSA_INCLUDE_DIRS}
  )
  set_target_properties(${target_name} PROPERTIES
    NO_SYSTEM_FROM_IMPORTED TRUE
  )

  if(TARGET mudnn_frontend)
    target_link_libraries(${target_name} PRIVATE
      mudnn_frontend
      ${MUSA_LIBRARIES}
    )
  else()
    message(FATAL_ERROR "mudnn_frontend target is required for frontend C++ tests")
  endif()
endfunction()

pybind11_add_module(mudnn_frontend_framework_test SHARED
  framework_graph_pybind.cc
)
mudnn_frontend_configure_test_target(mudnn_frontend_framework_test)

pybind11_add_module(mudnn_frontend_cpp_tests SHARED
  frontend_api_compat.cc
  frontend_api_builders.cc
  frontend_api_matmul.cc
  frontend_api_pointwise.cc
  frontend_api_graph_ops.cc
  frontend_api_sdpa.cc
  frontend_api_register.cc
)
mudnn_frontend_configure_test_target(mudnn_frontend_cpp_tests)

# Upstream cudnn-frontend test/cpp suite, ported to the mudnn frontend.
# Sources are kept under their upstream names for traceability; serialize.cpp
# documents the mudnn-specific deviations inline.
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.3.2
  )
  FetchContent_MakeAvailable(Catch2)
endif()

add_executable(mudnn_frontend_upstream_cpp_tests
  version.cpp
  validate.cpp
  tensor.cpp
  pointwise_tests.cpp
  get_engine_and_knobs.cpp
  serialize.cpp
)
mudnn_frontend_configure_test_target(mudnn_frontend_upstream_cpp_tests)
target_compile_options(mudnn_frontend_upstream_cpp_tests PRIVATE
  -Wno-unused-function
)
target_link_libraries(mudnn_frontend_upstream_cpp_tests PRIVATE
  Catch2::Catch2WithMain
)
add_test(NAME mudnn_frontend_upstream_cpp_tests
         COMMAND mudnn_frontend_upstream_cpp_tests)
