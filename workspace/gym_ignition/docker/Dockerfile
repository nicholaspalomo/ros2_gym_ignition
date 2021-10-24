### To build this docker container, run:
### docker build -t gym .
### docker build -t gym . --build-arg SSH_KEY="$(cat /path/to/.ssh/id_rsa)"

# ARG PARENT_IMAGE=nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
ARG PARENT_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
FROM ${PARENT_IMAGE}

# Enable OpenGl for ignition camera sensors (required to create context)
COPY --from=nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04 /usr/local /usr/local
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

### Use bash by default
SHELL ["/bin/bash", "-c"]

### Setup timezone (in order to prevent interactive install)
ARG TIMEZONE=America/Los_Angeles
ARG DEBIAN_FRONTEND=noninteractive
RUN rm -f /etc/localtime && \
    ln -s /usr/share/zoneinfo/"${TIMEZONE}" /etc/localtime && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
        locales \
        locales-all \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

### Install toolchain
ARG PYTHON3_VERSION=3.8
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        apt-utils \
        software-properties-common \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        ninja-build \
        libyaml-cpp-dev \
        gnupg2 \
        lsb-release \
        xauth \
        python3-pip \
        python3-wheel \
        python${PYTHON3_VERSION} \
        python${PYTHON3_VERSION}-dev \
        libpython${PYTHON3_VERSION}-dev \
        nano \
        libdart-dev && \
    rm -rf /var/lib/apt/lists/*

### Install ros2
ARG ROS2_DISTRO=foxy
ENV ROS2_DISTRO=${ROS2_DISTRO}
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
        ros-${ROS2_DISTRO}-ros-base \
        ros-${ROS2_DISTRO}-xacro && \
    rm -rf /var/lib/apt/lists/*

### Install ignition
ARG IGNITION_VERSION=edifice
ENV IGNITION_VERSION=${IGNITION_VERSION}
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list' && \
    wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
        ignition-${IGNITION_VERSION} && \
    rm -rf /var/lib/apt/lists/*

### Install torch
### Note: torch 1.8.0 does not always play nice with CUDA, restrict to 1.7.1 for now 
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        libopenmpi-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 --default-timeout=100 install \
        seaborn \
        optuna \
        stable-baselines3[extra] \
        sb3-contrib \
        gym \
        torch==1.7.1 \
        torchvision==0.8.2 \
        tqdm \
        yacs \
        ruamel.yaml \
        argcomplete

### Install Gym-Ignition
ARG GYM_IGNITION_DIR=/root/gym_ignition
ARG GYM_IGNITION_GIT_BRANCH=master
ENV GYM_IGNITION_DIR=${GYM_IGNITION_DIR}
ENV GYM_IGNITION_GIT_BRANCH=${GYM_IGNITION_GIT_BRANCH}

### Build iDynTree, fmt, Gym Ignition & ROS2 project
ARG DEPS_DIR=/root/deps
ENV DEPS_DIR=${DEPS_DIR}
RUN mkdir -p ${DEPS_DIR} && \
    cd ${DEPS_DIR} && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
    libeigen3-dev libxml2-dev coinor-libipopt-dev qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev qml-module-qtquick2 qml-module-qtquick-window2 qml-module-qtmultimedia qml-module-qtquick-dialogs qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel qml-module-qt-labs-settings && \
    git clone https://github.com/fmtlib/fmt.git && \
    mkdir -p ${DEPS_DIR}/fmt/build && \
    cd ${DEPS_DIR}/fmt/build && \
    cmake -DBUILD_SHARED_LIBS=TRUE .. && \
    make install -j2 && \
    rm -rf /var/lib/apt/lists/*

RUN cd ${DEPS_DIR} && \
    git clone https://github.com/robotology/idyntree.git && \
    mkdir -p ${DEPS_DIR}/idyntree/build && \
    cd ${DEPS_DIR}/idyntree/build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS:BOOL=OFF -DIDYNTREE_USES_PYTHON=False -DIDYNTREE_USES_IPOPT:BOOL=ON && \
    cmake --build . --target install -- -j2 && \
    rm -rf /var/lib/apt/lists/*

### Upgrade numpy (else iDynTree fails, not sure why)
RUN pip3 --default-timeout=100 install -U \
        numpy

### Add Github SSH key so that we can clone private gym_ignition repo (comment this out if public repo)
ARG SSH_KEY
ENV SSH_KEY=$SSH_KEY

# Make ssh dir (comment this out if public repo)
RUN mkdir /root/.ssh/ && \
    echo "$SSH_KEY" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    touch /root/.ssh/known_hosts && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone gym-ignition repo and install scenarIO library
RUN cd /root/ && \
    git clone git@github.com:nicholaspalomo/gym_ignition.git -b ${GYM_IGNITION_GIT_BRANCH} && \
    cd ${GYM_IGNITION_DIR}/workspace/scenario && \
    cmake -S . -B build/ && \
    cmake --build build/ --target install -- -j2 && \
    rm -rf /var/lib/apt/lists/*

### Install colcon, vcs toolkit and rosdep prior to building ROS-based projects
RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://repo.ros2.org/ubuntu/main $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    apt-get update && \
    apt-get install -yq --no-install-recommends \
        python3-colcon-common-extensions \
        python3-vcstool \
        python3-rosdep && \
    rosdep init && \
    rosdep update
# Note: Do not remove apt lists until last ROS-based project is built

### Set the paths for Ignition to find models and libraries
ENV IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:${GYM_IGNITION_DIR}/workspace/gym_ignition_description/gym_ignition_models:${GYM_IGNITION_DIR}/workspace/gym_ignition_description/worlds
ENV IGN_GAZEBO_PHYSICS_ENGINE_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/ign-physics-4/engine-plugins
ENV IGN_GAZEBO_SYSTEM_PLUGIN_PATH=${GYM_IGNITION_DIR}/workspace/build/lib:${GYM_IGNITION_DIR}/workspace/scenario/build/lib:$IGN_GAZEBO_PHYSICS_ENGINE_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:$IGN_GAZEBO_PHYSICS_ENGINE_PATH

### Add entrypoint sourcing the environment
COPY ./entrypoint.bash ./entrypoint.bash
RUN chmod +x ./entrypoint.bash

ENTRYPOINT ["/bin/bash", "entrypoint.bash"]
CMD ["/bin/bash"]