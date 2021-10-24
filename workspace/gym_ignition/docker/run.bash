#!/usr/bin/env bash

### Note - it is intended that this script be run on the host from the root folder of this repo with, e.g.:
### $ ./workspace/gym_ignition/docker/run.bash gym cartpole

### Note - need to have Nvidia Docker Toolkit installed (and have a GPU on your machine, of course) to uncomment the following:
# # Docker
# curl https://get.docker.com | sh \
#   && sudo systemctl --now enable docker
# # Nvidia Docker
# distribution=$(. /etc/os-release; echo $ID$VERSION_ID) \
#   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
#   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# sudo apt-get update && sudo apt-get install -y nvidia-docker2
# sudo systemctl restart docker

if [ $# -lt 1 ]; then
    echo "Usage: $0 <docker image>"
    exit 1
fi

IMG=$1
ARGS=("$@")

# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
    if [ ! -z "$xauth_list" ]; then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

DOCKER_OPTS=""

# # Get the current version of docker-ce
# # Strip leading stuff before the version number so it can be compared
# DOCKER_VER=$(dpkg-query -f='${Version}' --show docker-ce | sed 's/[0-9]://')
# if dpkg --compare-versions 19.03 gt "$DOCKER_VER"; then
#     echo "Docker version is less than 19.03, using nvidia-docker2 runtime"
#     if ! dpkg --list | grep nvidia-docker2; then
#         echo "Please either update docker-ce to a version greater than 19.03 or install nvidia-docker2"
#         exit 1
#     fi
#     DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
# else
#     DOCKER_OPTS="$DOCKER_OPTS --gpus all"
# fi

# Prevent executing "docker run" when xauth failed.
if [ ! -f $XAUTH ]; then
    echo "[$XAUTH] was not properly created. Exiting..."
    exit 1
fi

docker run -it \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "$XAUTH:$XAUTH" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/etc/localtime:/etc/localtime:ro" \
    -v "/dev/input:/dev/input" \
    -v "$PWD:/root/gym_ignition" \
    $ADDITIONAL_VOLUMES \
    --network host \
    --ipc host \
    --rm \
    -it \
    --privileged \
    --security-opt seccomp=unconfined \
    $DOCKER_OPTS \
    $IMG \
    ${@:2}

    # Use this volume for custom config and models that are stored locally on your machine
    # -v "$HOME/.ignition:/root/.ignition"