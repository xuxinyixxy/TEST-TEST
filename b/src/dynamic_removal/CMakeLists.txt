cmake_minimum_required(VERSION 3.8)
project(dynamic_removal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL 1.12 REQUIRED COMPONENTS common io filters kdtree)

include_directories( include ${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(dynamic_removal_node src/dynamic_removal.cpp)

target_link_libraries(dynamic_removal_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS dynamic_removal_node DESTINATION lib/${PROJECT_NAME})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_target_dependencies(dynamic_removal_node
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
)

ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
)

ament_export_include_directories(include)
ament_export_libraries(dynamic_removal_node)
ament_package()