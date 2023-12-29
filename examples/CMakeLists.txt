cmake_minimum_required(VERSION 3.14)

project(LazyExamples CXX)

include(FetchContent)
FetchContent_Declare(Lazy SOURCE_DIR "${PROJECT_SOURCE_DIR}/..")
FetchContent_MakeAvailable(Lazy)

foreach(name IN ITEMS example-1 example-2 example-3 example-mergesort)
  add_executable("${name}" "${name}.cc")
  target_link_libraries("${name}" PRIVATE Lazy::Lazy)
  target_compile_features("${name}" PRIVATE cxx_std_17)
endforeach()
