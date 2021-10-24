cmake_minimum_required(VERSION 3.16)
project(gym_ignition_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY gym_ignition_models worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")

ament_package()