# libeclipse — native core for EclipseBackend (see docs/CPP_ROADMAP.md).
#
# Targets:
#   cspice     static, C89, vendored NAIF CSPICE N0067  (third_party/cspice)
#   erfa       static, C,   vendored liberfa 2.0.1       (third_party/erfa)
#   eclipse    static, C++20, the core library           (core/)
#   _eclipse   nanobind extension module                 (bindings/)   [ECLIPSE_BUILD_PYTHON]
#   eclipse_tests  Catch2 unit + parity tests            (tests/cpp/)  [ECLIPSE_BUILD_TESTS]
cmake_minimum_required(VERSION 3.20)
project(libeclipse LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # everything ends up inside a Python extension
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ECLIPSE_BUILD_PYTHON "Build the nanobind extension module _eclipse" ON)
option(ECLIPSE_BUILD_TESTS  "Build the Catch2 test executable"             OFF)
option(ECLIPSE_SANITIZE     "ASan+UBSan on libeclipse only (never on CSPICE)" OFF)
option(ECLIPSE_OPENMP       "OpenMP for circumstances_grid (serial fallback when absent)" ON)

# ccache when available (CSPICE is ~2,200 translation units; first build 2-4 min).
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# No -ffast-math anywhere: bit-level parity with the Python oracle (roadmap §7).
# -ffp-contract=off keeps FMA contraction from producing 1-ulp parity drift.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(ECLIPSE_FP_FLAGS -ffp-contract=off)
elseif(MSVC)
  set(ECLIPSE_FP_FLAGS /fp:precise)
endif()

include(cmake/cspice.cmake)
include(cmake/erfa.cmake)

# ---------------------------------------------------------------- libeclipse
add_library(eclipse STATIC
  core/src/deltat.cpp
  core/src/eop.cpp
  core/src/ephem.cpp
  core/src/geometry.cpp
  core/src/circumstances.cpp
  core/src/catalog.cpp
  core/src/limb.cpp
  core/src/besselian.cpp
)
target_include_directories(eclipse PUBLIC core/include)
target_link_libraries(eclipse PUBLIC cspice erfa)
target_compile_options(eclipse PRIVATE ${ECLIPSE_FP_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(eclipse PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()
if(ECLIPSE_SANITIZE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(eclipse PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(eclipse PUBLIC -fsanitize=address,undefined)
endif()

# OpenMP for the observer loop of circumstances_grid (roadmap §5 phase 3). The
# elements are evaluated once under the SPICE lock; the per-observer math is
# pure and lock-free, so the loop parallelizes without touching CSPICE. Never
# required: Apple Clang without libomp (and any other toolchain lacking it)
# builds the serial loop, selected in the source by ECLIPSE_HAVE_OPENMP.
# -fopenmp does not alter ECLIPSE_FP_FLAGS (-ffp-contract=off stays in force).
if(ECLIPSE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(eclipse PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(eclipse PUBLIC ECLIPSE_HAVE_OPENMP)
    message(STATUS "libeclipse: OpenMP ${OpenMP_CXX_VERSION} found; circumstances_grid is parallel")
  else()
    message(STATUS "libeclipse: OpenMP not found; circumstances_grid builds serial")
  endif()
else()
  message(STATUS "libeclipse: ECLIPSE_OPENMP=OFF; circumstances_grid builds serial")
endif()

# ------------------------------------------------------------------ bindings
if(ECLIPSE_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  # nanobind is a build dependency in pyproject.toml; locate its CMake config.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_eclipse NB_STATIC
    bindings/_eclipse.cpp
    bindings/bind_ellipsoid.cpp
    bindings/bind_geometry.cpp
    bindings/bind_circumstances.cpp
    bindings/bind_catalog.cpp
    bindings/bind_limb.cpp
    bindings/bind_besselian.cpp
  )
  target_link_libraries(_eclipse PRIVATE eclipse)
  target_compile_options(_eclipse PRIVATE ${ECLIPSE_FP_FLAGS})
  install(TARGETS _eclipse LIBRARY DESTINATION .)
endif()

# --------------------------------------------------------------------- tests
if(ECLIPSE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
