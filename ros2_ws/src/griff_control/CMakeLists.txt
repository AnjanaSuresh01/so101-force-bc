cmake_minimum_required(VERSION 3.16)
project(griff_control)

if(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(controller_interface REQUIRED)
find_package(griff_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)

set(THIS_PACKAGE_DEPENDS
  controller_interface
  griff_msgs
  hardware_interface
  kdl_parser
  orocos_kdl_vendor
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
)

add_library(${PROJECT_NAME} SHARED src/force_limited_admittance_controller.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
ament_target_dependencies(${PROJECT_NAME} PUBLIC ${THIS_PACKAGE_DEPENDS})

pluginlib_export_plugin_description_file(controller_interface griff_control_plugins.xml)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Deliberately no ament_lint_auto. The style linters (uncrustify, cpplint,
  # copyright) cannot be run on the machine this was written on, so enabling
  # them would mean shipping a CI job whose failures nobody here can reproduce.
  # What CI does check is that the controller builds and that its force bound
  # holds -- which is the part that matters.
  # The admittance maths is header-only so this test needs neither a node nor a
  # controller manager -- it is the same closed-loop wall model the Python suite
  # uses, asserting the same bound.
  ament_add_gtest(test_admittance_core test/test_admittance_core.cpp)
  target_include_directories(test_admittance_core PRIVATE include)
  target_link_libraries(test_admittance_core Eigen3::Eigen)
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_DEPENDS})
ament_package()
