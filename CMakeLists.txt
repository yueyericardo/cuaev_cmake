cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(test_model)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
add_subdirectory(cuaev)

add_executable(test_model test_model.cpp)
target_link_libraries(test_model ${TORCH_LIBRARIES})
target_link_libraries(test_model ${PYTHON_LIBRARIES})
target_link_libraries(test_model -Wl,--no-as-needed cuaev)
set_property(TARGET test_model PROPERTY CXX_STANDARD 14)
