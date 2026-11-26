cmake_minimum_required(VERSION 3.12)

project(mudnn_frontend LANGUAGES CXX)

include(CTest)

set(MUDNN_FRONTEND_INSTALL_PYTHON_DIR "python" CACHE STRING
    "Install destination for mudnn-frontend Python packages")
option(MUDNN_FRONTEND_BUILD_FRAMEWORK_TESTS
       "Build standalone mudnn_frontend framework pytest bindings"
       OFF)

set(_MUDNN_FRONTEND_DEFAULT_BACKEND_ROOT "${CMAKE_CURRENT_LIST_DIR}/muDNN/build/export/mudnn")
if(NOT EXISTS "${_MUDNN_FRONTEND_DEFAULT_BACKEND_ROOT}")
  set(_MUDNN_FRONTEND_DEFAULT_BACKEND_ROOT "/usr/local/musa")
endif()

set(MUDNN_BACKEND_ROOT "$ENV{MUDNN_BACKEND_ROOT}" CACHE PATH
    "muDNN backend installation or staged build root")
if(NOT MUDNN_BACKEND_ROOT)
  set(MUDNN_BACKEND_ROOT "${_MUDNN_FRONTEND_DEFAULT_BACKEND_ROOT}" CACHE PATH
      "muDNN backend installation or staged build root" FORCE)
endif()
set(MUDNN_BACKEND_INCLUDE "$ENV{MUDNN_BACKEND_INCLUDE}" CACHE PATH
    "muDNN backend include directory")
if(NOT MUDNN_BACKEND_INCLUDE)
  set(MUDNN_BACKEND_INCLUDE "${MUDNN_BACKEND_ROOT}/include" CACHE PATH
      "muDNN backend include directory" FORCE)
endif()
set(MUDNN_BACKEND_LIB "$ENV{MUDNN_BACKEND_LIB}" CACHE PATH
    "muDNN backend library directory")
if(NOT MUDNN_BACKEND_LIB)
  set(MUDNN_BACKEND_LIB "${MUDNN_BACKEND_ROOT}/lib" CACHE PATH
      "muDNN backend library directory" FORCE)
endif()

if(NOT TARGET mudnn)
  find_library(MUDNN_BACKEND_LIBRARY
    NAMES mudnn libmudnn.so.3
    HINTS "${MUDNN_BACKEND_LIB}" "${MUDNN_BACKEND_ROOT}/lib64"
  )
  if(NOT MUDNN_BACKEND_LIBRARY)
    message(FATAL_ERROR
      "libmudnn was not found. Set MUDNN_BACKEND_ROOT or "
      "MUDNN_BACKEND_LIB to a muDNN backend installation.")
  endif()
  add_library(mudnn_backend SHARED IMPORTED GLOBAL)
  set_target_properties(mudnn_backend PROPERTIES
    IMPORTED_LOCATION "${MUDNN_BACKEND_LIBRARY}"
  )
  if(EXISTS "${MUDNN_BACKEND_INCLUDE}")
    set_target_properties(mudnn_backend PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${MUDNN_BACKEND_INCLUDE}"
    )
  endif()
endif()

add_library(mudnn_frontend INTERFACE)
target_include_directories(mudnn_frontend INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
if(EXISTS "${MUDNN_BACKEND_INCLUDE}")
  target_include_directories(mudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${MUDNN_BACKEND_INCLUDE}>
  )
endif()
if(RUNTIME_INCLUDE)
  target_include_directories(mudnn_frontend INTERFACE ${RUNTIME_INCLUDE})
endif()
if(TARGET mudnn)
  target_link_libraries(mudnn_frontend INTERFACE mudnn)
else()
  target_link_libraries(mudnn_frontend INTERFACE mudnn_backend)
endif()
target_compile_features(mudnn_frontend INTERFACE cxx_std_17)

if(MUDNN_FRONTEND_BUILD_FRAMEWORK_TESTS)
  add_subdirectory(tests/cpp)
endif()

install(TARGETS mudnn_frontend EXPORT mudnnTargets)
install(DIRECTORY include/ DESTINATION include)
install(CODE
  "file(REMOVE_RECURSE
     \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MUDNN_FRONTEND_INSTALL_PYTHON_DIR}/mudnn_frontend/__pycache__\"
     \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MUDNN_FRONTEND_INSTALL_PYTHON_DIR}/mudnn/__pycache__\"
     \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MUDNN_FRONTEND_INSTALL_PYTHON_DIR}/cudnn_frontend/__pycache__\")")
install(DIRECTORY python/mudnn python/mudnn_frontend python/cudnn_frontend
        DESTINATION ${MUDNN_FRONTEND_INSTALL_PYTHON_DIR}
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)
