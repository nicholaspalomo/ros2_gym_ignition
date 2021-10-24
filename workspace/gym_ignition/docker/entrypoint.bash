#!/usr/bin/env bash
set -e

cd ${GYM_IGNITION_DIR}

source /opt/ros/${ROS2_DISTRO}/setup.bash

MAKEFLAGS=-j1 colcon build --parallel-workers 1 --cmake-args -DENV_NAME=$1 

source install/setup.bash

ros2 launch gym_ignition gym_ignition.launch.py env:=$1 cfg:=cfg.yaml render:=True