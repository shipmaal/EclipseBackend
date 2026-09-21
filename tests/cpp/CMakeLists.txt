# Catch2 v3 via FetchContent (GitHub is reachable where NAIF is not, roadmap §6).
include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(Catch2)

add_executable(eclipse_tests
  test_ephem.cpp test_deltat.cpp test_elements.cpp
  test_numerics.cpp test_ellipsoid.cpp test_geometry.cpp test_circumstances.cpp
  test_catalog.cpp
)
target_link_libraries(eclipse_tests PRIVATE eclipse Catch2::Catch2WithMain)
target_compile_options(eclipse_tests PRIVATE ${ECLIPSE_FP_FLAGS})
# Kernels live in <repo>/kernels (gitignored; `uv run python -m kernels.bootstrap`).
target_compile_definitions(eclipse_tests PRIVATE
  ECLIPSE_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels"
  ECLIPSE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(eclipse_tests)
