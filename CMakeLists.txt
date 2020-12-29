cmake_minimum_required(VERSION 3.14)

project(Lazy
        VERSION 0.0.1
        DESCRIPTION "Parallel function calls and continuations in C++ based on Eric Niebler's talk at CppCon 2019"
        HOMEPAGE_URL "https://github.com/tirimatangi/Lazy"
        LANGUAGES CXX)

# ---- Warning guard ----

# Protect dependents from this project's warnings if the guard isn't disabled
set(Lazy_warning_guard SYSTEM)
if(Lazy_INCLUDE_WITHOUT_SYSTEM)
  set(Lazy_warning_guard "")
endif()

# ---- Declare library ----

add_library(Lazy INTERFACE)
add_library(Lazy::Lazy ALIAS Lazy)

target_include_directories(Lazy
        ${Lazy_warning_guard}
        INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

target_compile_features(Lazy INTERFACE cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(Lazy INTERFACE Threads::Threads)

# ---- Install ----

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(Lazy_directory "Lazy-${PROJECT_VERSION}")
set(Lazy_include_directory
        "${CMAKE_INSTALL_INCLUDEDIR}/${Lazy_directory}")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${Lazy_include_directory}")

install(TARGETS Lazy
        EXPORT LazyTargets
        INCLUDES DESTINATION "${Lazy_include_directory}")

write_basic_package_version_file(
        LazyConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

set(Lazy_install_cmakedir
        "${CMAKE_INSTALL_LIBDIR}/cmake/${Lazy_directory}")

install(FILES
        "${PROJECT_SOURCE_DIR}/cmake/LazyConfig.cmake"
        "${PROJECT_BINARY_DIR}/LazyConfigVersion.cmake"
        DESTINATION "${Lazy_install_cmakedir}")

install(EXPORT LazyTargets
        NAMESPACE Lazy::
        DESTINATION "${Lazy_install_cmakedir}")
