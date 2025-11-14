cmake_minimum_required(VERSION 3.15)
project(trace_target)

# [global] # 全局参数
# # 选项：
# # - compile_chassis
# # - load_chassis
set(chassis_compile_command "load_chassis")

# [global] # 全局参数
# # 选项：
# # - true
# # - false
set(graphic_enable "true")

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${Eigen3_INCLUDE_DIRS})

include_directories(.)
add_subdirectory(kalman)

if (${graphic_enable} STREQUAL "true")
    add_compile_definitions(CMAKE_DEF_WITH_OPENCV)
endif()
add_subdirectory(graphic)

if (${chassis_compile_command} STREQUAL "compile_chassis")
    add_subdirectory(chassis)
elseif (${chassis_compile_command} STREQUAL "load_chassis")
    link_directories(lib)
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} Ceres::ceres chassis_lib display_board)
