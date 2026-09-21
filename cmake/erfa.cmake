# Vendored liberfa 2.0.1 (the version bundled by pyerfa 2.0.1.x, so ERFA
# results are bit-identical between the C++ core and the Python oracle).
# The upstream tree is untouched; the version macros that autotools would put
# in config.h are generated here from cmake/erfa_config.h.in.
set(ERFA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/erfa")
file(GLOB ERFA_SOURCES CONFIGURE_DEPENDS "${ERFA_DIR}/src/*.c")

set(ERFA_VERSION "2.0.1")
set(ERFA_VERSION_MAJOR 2)
set(ERFA_VERSION_MINOR 0)
set(ERFA_VERSION_MICRO 1)
set(ERFA_SOFA_VERSION "20231011")   # from erfa-2.0.1/configure.ac
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/erfa_config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/erfa_config/config.h" @ONLY)

add_library(erfa STATIC ${ERFA_SOURCES})
target_include_directories(erfa PUBLIC "${ERFA_DIR}/src")
target_include_directories(erfa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/erfa_config")
target_compile_definitions(erfa PRIVATE HAVE_CONFIG_H)
set_target_properties(erfa PROPERTIES C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(erfa PRIVATE -O2 -w -ffp-contract=off)
endif()
target_link_libraries(erfa PUBLIC m)
