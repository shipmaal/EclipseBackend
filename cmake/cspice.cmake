# Vendored NAIF CSPICE N0067, compiled as C in its own target — never as C++.
#
# Flags are NAIF's own for PC_Linux_GCC_64bit, from third_party/cspice/mkprodct.csh
# ("-c -ansi -m64 -O2 -fPIC -DNON_UNIX_STDIO"); MSVC flags from the PC_Windows
# mkprodct.bat. Nothing else is added: no -ffast-math, no sanitizers (the f2c
# output trips UBSan constantly), warnings silenced (-w) because this is not our
# code to fix. See docs/CPP_ROADMAP.md §2 and THIRD_PARTY_NOTICES.md.
set(CSPICE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cspice")
file(GLOB CSPICE_SOURCES CONFIGURE_DEPENDS "${CSPICE_DIR}/src/*.c")

add_library(cspice STATIC ${CSPICE_SOURCES})
target_include_directories(cspice PUBLIC "${CSPICE_DIR}/include")
set_target_properties(cspice PROPERTIES
  C_STANDARD 90
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF          # == -ansi
  POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_definitions(cspice PRIVATE MSDOS NON_ANSI_STDIO _COMPLEX_DEFINED OMIT_BLANK_CC)
  target_compile_options(cspice PRIVATE /O2 /w)
else()
  target_compile_definitions(cspice PRIVATE NON_UNIX_STDIO)
  target_compile_options(cspice PRIVATE -O2 -w)
  # -ansi hides some POSIX declarations the f2c I/O layer uses; NAIF's build
  # tolerates the implicit declarations, GCC 14+ / recent Clang do not.
  target_compile_options(cspice PRIVATE -Wno-implicit-function-declaration -Wno-error=implicit-function-declaration)
endif()
target_link_libraries(cspice PUBLIC m)
