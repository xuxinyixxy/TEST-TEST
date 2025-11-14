cmake_minimum_required(VERSION 3.15)

if (${graphic_enable} STREQUAL "true")
    find_package(OpenCV 4 REQUIRED)
endif()

include_directories(.)

add_library(display_board display_board.cpp)
if (${graphic_enable} STREQUAL "true")
    target_link_libraries(display_board ${OpenCV_LIBS})
    target_include_directories(display_board PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()