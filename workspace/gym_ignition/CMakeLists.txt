cmake_minimum_required(VERSION 3.16)
project(gym_ignition)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED) # https://fmt.dev/latest/usage.html#usage-with-cmake
find_package(iDynTree REQUIRED) # Inverse kinematics library
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rmw_implementation_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED) # for generating custom msgs, srvs
find_package(Scenario COMPONENTS Gazebo REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed -w -funroll-loops -pthread")

message(STATUS "[gym_ignition] prefix path ${CMAKE_PREFIX_PATH}")
message(STATUS "[gym_ignition] including directory ${GYM_IGN_ENVIRONMENT_INCLUDE_PATH}")

# Include the environment you want to train
set(INCLUDE_DIR
  ${INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${GYM_IGN_ENVIRONMENT_INCLUDE_PATH}
  include/${PROJECT_NAME}
  include/${PROJECT_NAME}/${ENV_NAME})

# Build and install the custom service headers
rosidl_generate_interfaces(${PROJECT_NAME}
    "srv/Info.srv"
    "srv/Reset.srv"
    "srv/Step.srv"
    "srv/Extra.srv"
    "srv/Observe.srv"
    "msg/Depth.msg"
    "msg/Rgb.msg"
    "msg/Thermal.msg"
    DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install service files
install(DIRECTORY srv
    DESTINATION share/${PROJECT_NAME})

# Make a debug executable, depending on the value of DEBUG provided
# Make sure to build with (example):
# colcon build --packages-select gym_ignition --cmake-args -DENV_NAME=panda_grasp -DCMAKE_BUILD_TYPE=RelWithDebInfo
if(DEBUG STREQUAL "true" OR DEBUG STREQUAL "TRUE" OR DEBUG STREQUAL "True")
  message(STATUS "[gym_ignition] BUILDING THE DEBUG EXECUTABLE")
  add_executable(env src/debug.cpp)
else()
  # Build the environment executable
  add_executable(env src/gym_ignition.cpp)
endif()

get_available_rmw_implementations(rmw_implementations2)
foreach(rmw_implementation ${rmw_implementations2})
  find_package("${rmw_implementation}" REQUIRED)
  get_rmw_typesupport(typesupport_impls "${rmw_implementation}" LANGUAGE "cpp")
  foreach(typesupport_impl ${typesupport_impls})
    rosidl_target_interfaces(env
      ${PROJECT_NAME} ${typesupport_impl}
    )
  endforeach()
endforeach()

# Set libraries to link against executable
set(LIBRARIES
  fmt::fmt-header-only
  ${iDynTree_LIBRARIES}
  ScenarioGazebo::ScenarioGazebo
  ScenarioGazebo::GazeboSimulator
  yaml-cpp)

target_link_libraries(env ${LIBRARIES})
ament_target_dependencies(env rclcpp)

target_include_directories(env PUBLIC ${INCLUDE_DIR})
target_compile_options(env PUBLIC -mtune=native -fPIC -O3)

message(STATUS "[gym_ignition] CMAKE INSTALL PREFIX ${CMAKE_INSTALL_PREFIX}")
message(STATUS "[gym_ignition] EXECUTABLES INSTALLED TO ${CMAKE_INSTALL_BINDIR}")
message(STATUS "[gym_ignition] LIBRARIES INSTALLED TO ${CMAKE_INSTALL_LIBDIR}")

# Install cpp environment node
install(TARGETS
  env
  DESTINATION lib/${PROJECT_NAME})

# Install all Python modules under the gym_ignition directory
ament_python_install_package(${PROJECT_NAME})

# Install the Python vecenv module
install(PROGRAMS
  scripts/${ENV_NAME}.py
  DESTINATION lib/${PROJECT_NAME})

# Install the launch file
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME})

# Additional Python modules
install(DIRECTORY
  gym_ignition
  DESTINATION share/${PROJECT_NAME})

# Header file containing environment config
install(DIRECTORY
  include/gym_ignition/${ENV_NAME}
  DESTINATION share/${PROJECT_NAME})

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")

ament_package()