# ROS2/Ignition Gazebo Environment for Reinforcement Learning

** This is a work in progress! **

This project builds upon the original [`gym-ignition` project](https://github.com/robotology/gym-ignition) started by the team at the Italian Institute of Technology and is also heavily inspired by the [`RaiSim Gym`](https://raisim.com/index.html) project from ETH Zurich. However, this gym environment makes use of ROS2 to run multiple, parallel instances of your robot in order to accelerate the training/learning of RL policies in simulation. Where possible, this project tries to adhere to an API similar to that of [OpenAI Gym](https://github.com/openai/gym).

## TL;DR

If you want to try out the project without installing all the dependencies, run the Dockerfile in the `~/gym_ignition/workspace/gym_ignition/docker` directory.

First, make sure to have a working [Docker](https://docs.docker.com/engine/install/ubuntu/) installation on your system.

To build the Dockerfile, run:

```
docker build -t gym .
```

To launch the simulation from the Dockerfile, from a clean colcon workspace, run:

```
./workspace/gym_ignition/docker/run.bash gym cartpole
```
for the cartpole example.

## Installation

To start training policies right away, do the following steps:

1. If you want to try out the project without installing the dependencies, you can build and run the Docker project by following the instructions in the previous section.

2. You need to install the following libraries and dependencies:

    - [ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
    - [iDynTree](https://github.com/robotology/idyntree#installation) - used for computations of forward/inverse kinematics and dynamics. NOTE: Make sure to compile with flag `-DIDYNTREE_USES_IPOPT:BOOL=ON`!
    - [Ignition Edifice](https://ignitionrobotics.org/docs/edifice/install_ubuntu) - the rigid body physics simulator used in this project
    - [fmt](https://fmt.dev/latest/usage.html#installing-the-library) - for `"".format`-like formatting of string expressions in C++
    - other dependencies - 
    
    ```
    sudo apt install libeigen3-dev libsdformat6-dev libyaml-cpp-dev python3-vcstool python3-colcon-common-extensions && pip3 install ruamel.yaml gym stable-baselines3 opencv-python tensorboard
    ```

3. Clone this repo to your machine.

4. Build the ScenarIO libraries:

```
cd ~/gym_ignition/workspace/scenario
cmake -S . -B build/
sudo cmake --build build/ --target install
```

5. To build your code, in your colcon workspace, run:

```
colcon build --cmake-args -DENV_NAME=cartpole
```

This code will build the cartpole demo.

6. Source your colcon workspace, e.g.

```
source install/setup.bash
```

7. You need to set several paths in order that Ignition finds the meshes and libraries:

```
IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$GYM_IGNITION_DIR/workspace/gym_ignition_description/gym_ignition_models:$GYM_IGNITION_DIR/workspace/gym_ignition_description/worlds

IGN_GAZEBO_PHYSICS_ENGINE_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/ign-physics-4/engine-plugins

IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$GYM_IGNITION_DIR/workspace/build/lib:$GYM_IGNITION_DIR/workspace/scenario/build/lib:$IGN_GAZEBO_PHYSICS_ENGINE_PATH

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:$IGN_GAZEBO_PHYSICS_ENGINE_PATH
```
where `GYM_IGNITION_DIR` is an environment variable containing the **absolute** path to this repository. It is recommended that you append these paths to your bashrc.

**NOTE**: You may want to check out [this](https://github.com/ros2/ros2/issues/451) issue on GitHub if you have both ROS1 and ROS2 installed on your system and modify your `PYTHONPATH` accordingly for ROS2.

8. To launch your training session, run:

```
ros2 launch gym_ignition gym_ignition.launch.py env:=cartpole cfg:=cfg.yaml render:=True
```

which will launch the training for the cartpole demo. Note: To run any simulation in headless mode, simply change the `render` argument to `False`. This may give you a slight speed increase during training.

- For the PPO and the SAC algorithms, a Tensorboard window will appear upon launching the training, allowing you to monitor the progress of your agent in real-time.

## Some notes for helping you get up and running with training your own RL agent

- Check out the repository structure below. This will give you an idea of the different relevant portions of the repo.

- Let's assume you want to train a humanoid to walk using RL. You call this training scenario by the codename `humanoid`. To make your own MDP, you **must**, at a minimum, write/implement the following files:

    - A YAML config file containing the relevant parameters for training your agent. You should put this config file in the `config` directory in a folder called `humanoid`.
    - A runner script called `humanoid.py` in the `scripts` directory in order to launch the training.
    - A header file named `Environment.hpp` in a folder called `humanoid` inside `include/gym_ignition`. _This is where you actually implement the logic for your MDP and is the most important file in your simulation._

- To see a complete, working example, tuned for the PPO and the SAC policy optimization algorithms, check out these three files for the cartpole example.

- As your agent is training, the neural network parameters will be backed-up in the `trained_params` folder of this repo.

- You also can use a camera to get photorealistic renderings to use in your training. To get you started on using a camera with `gym_ignition`, checkout `Camera.hpp`, `panda_cnn/Environment.hpp`, `config/panda_cnn/cfg.yaml`, and `gym_ignition_description/gym_ignition_models/kinect/kinect.xacro`. Checkout the various methods implemented in `GymIgnitionVecEnv.py` and also in `ppo_cnn.py` to see how to access the camera feed in Python.

## Repository Structure

```bash
├── trained_params                      # Neural network parameters (PyTorch)
└── workspace                           # Main folder containing source code for the project
    ├── gym_ignition                    # Robot URDFs and meshes
        ├── env-hooks                   # Environment hooks, so that Ignition Gazebo knows where to find ScenarIO libraries
        ├── config                      # YAML config files for training RL agents, setting up simulation
            └── cartpole                # YAML config file for cartpole example
        ├── docker                      # Docker files for running this project, e.g. on a cluster (WARNING: These are probably out of date)
        ├── gym_ignition                # Python Vectorized Environment node implementation
            ├── algo                    # Policy optimization algorithms
                ├── dagger              # Implementation of DAgger (Dataset Aggregation)
                ├── ddpg                # Implementation of DDPG (Deep Deterministic Policy Gradient)
                ├── ppo                 # Implementation of PPO (Proximal Policy Optimization)
                ├── sac                 # Implementation of SAC (Soft Actor-Critic)
                ├── tcn                 # TCN (Temporal Convolutional Network) implementation
                └── td3                 # Implementation of TD3 (Twin-Delayed DDPG)    
            └── helper                  # Helper scripts
        ├── include                     # Header files containing the MDPs for training RL agents
            └── gym_ignition
                └── cartpole            # Header file for cartpole example
        ├── launch                      # Master launch file
        ├── msg                         # Message definitions for camera images
        ├── scripts                     # Runner scripts for launching Python node
        ├── src                         # C++ node source code
        └── srv                         # Service definitions
    ├── gym_ignition_description        # Robot URDFs and meshes
        ├── env-hooks                   # Environment hooks, so that Ignition Gazebo knows where to find the mesh, world files
        ├── gym_ignition_models         # Robot URDFs and meshes
        └── worlds                      # World SDF files
    └── scenario                        # Source code for ScenarIO - see https://github.com/robotology/gym-ignition/tree/master/scenario
```

## Debugging

In developing this code, I used Visual Studio Code as my preferred IDE. You can download it here: [https://code.visualstudio.com/](https://code.visualstudio.com/).

### Python

Debugging ROS nodes in Python is a little ugly. You can, however, checkout the instructions given here for VS Code: https://github.com/ms-iot/vscode-ros/blob/master/doc/debug-support.md

### C++

Debugging ROS nodes in C++ is very convenient. To debug your MDP (i.e. code in `Environment.hpp`), you should do the following:

0. Clean your colcon workspace! Otherwise weird things might happen when launching the ROS node built with debug symbols.
1. In VS Code, you're gonna want to create a [debug configuration](https://code.visualstudio.com/docs/editor/debugging) for your application in your `launch.json` file. An example:

```
"configurations": [
      {
          "name": "(gdb) Launch gym_ignition debug app",
          "type": "cppdbg",
          "request": "launch",
          "program": "${workspaceFolder}/install/gym_ignition/lib/gym_ignition/env",
          "args": [
              "${workspaceFolder}/install/gym_ignition/share/gym_ignition/config/cartpole/cfg.yaml"
          ],
          "logging": { "engineLogging": true },
          "stopAtEntry": true,
          "cwd": "${workspaceFolder}/workspace",
          "environment": [
              {
                  "name": "LD_LIBRARY_PATH",
                  "value": "${LD_LIBRARY_PATH}:${workspaceFolder}/install/gym_ignition/lib:/opt/ros/foxy/lib"
              }
          ],
          "externalConsole": false,
          "MIMode": "gdb",
          "setupCommands": [
              {
                  "description": "Enable pretty-printing for gdb",
                  "text": "-enable-pretty-printing",
                  "ignoreFailures": true
              }
          ]
      }
  ]
```

where, here, your executable is named `env`.

2. Check out the source code for the debug node in `gym_ignition/src/debug.cpp` and adjust the node's source code as necessary for your application.

3. Place breakpoints at the relevant points of interest in your code (just to the left of the line numbers).

4. Build the executable using the flag to build the node with debug symbols:

```
colcon build --cmake-args -DENV_NAME=cartpole -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Note: It's recommended that you modify your `cfg.yaml` so that you don't launch multiple environments when launching the debug executable (i.e. change `num_envs` parameter to 1). 

5. Source your workspace (as always):

```
source install/setup.bash
```

6. Open the `Run and Debug` menu on the left-hand side of VS Code. Here, after selecting your launch configuration at the top of the panel, press the green "run" button to start the debugging your code, during which time you can inspect the stack variables at the breakpoint locations.

### Docker

1. To build the Docker image, navigate to the `docker` directory and run:

```
docker build -t gym . --build-arg SSH_KEY="$(cat /absolute/path/to/.ssh/id_rsa)"
```
Note that you may first need to comment out the entrypoint command at the bottom of the Dockerfile.

2. The terminal output will indicate the ID of the built image, e.g. `889e3db8e14a`. Start this image with:

```
docker run -it -d 889e3db8e14a  /bin/bash
```

3. Next, get the container ID with:

```
docker ps -a
```

Most likely, the first entry that appears is your newly-launched Docker container. Its container ID should have been automatically assigned a random name, such as `agitated_gould`. To attached to your running Docker container, run:

```
docker exec -it agitated_gould /bin/bash
```

in the terminal. From here, you can browse around the Docker container, build, and launch the code.

7. To kill the docker instance, run:

```
docker kill agitated_gould
```
or specify as the third argument whatever the name of your Docker container may be.
## Contributors
Nicholas Palomo, ETH Zurich (njpalomo@outlook.com // https://www.linkedin.com/in/nicholaspalomo/). Feel free to reach out and connect on LinkedIn! I'd also be happy to answer any questions about the code in this repo, as I may have forgotten some details as I wrote this ReadMe in a hurry. 😛 Thanks for stopping by!