import os
import launch_ros
import launch
from launch.substitutions import LaunchConfiguration
from ruamel.yaml import YAML, dump, RoundTripDumper

def generate_launch_actions(parameter_file_path, environment_file_path, env_name, render):
    # Load the configuration file
    cfg = YAML().load(open(parameter_file_path, 'r'))

    # Launch num_env nodes
    env_node_launch_descriptions = []
    for i in range(cfg['environment']['num_envs']):
        env_node_launch_descriptions.append(
            launch_ros.actions.Node(
                package='gym_ignition',
                executable='env',
                name='env{}'.format(i),
                output='log',
                arguments=[parameter_file_path, dump(cfg['environment'], Dumper=RoundTripDumper), '{}'.format(i), render]
            )
        )

    trainer_node_launch_description = launch_ros.actions.Node(
        package='gym_ignition',
        executable=env_name + '.py',
        name='vecenv',
        output='screen',
        arguments=['cfg', parameter_file_path, 'env', environment_file_path]
    )

    env_node_launch_descriptions.append(trainer_node_launch_description)
    return env_node_launch_descriptions

def opaque_func(context, *args, **kwargs):

    pkg_share = launch_ros.substitutions.FindPackageShare(package='gym_ignition').find('gym_ignition')

    env_name = LaunchConfiguration('env').perform(context)
    cfg_name = LaunchConfiguration('cfg').perform(context)
    render   = LaunchConfiguration('render').perform(context)

    parameter_file_path = os.path.join(pkg_share, 'config', env_name, cfg_name)

    environment_file_path = os.path.join(pkg_share, env_name, 'Environment.hpp')

    return generate_launch_actions(parameter_file_path, environment_file_path, env_name, render)

def generate_launch_description():

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='env',
            default_value='panda',
            description='name of the environment (experiment) to launch'
        ),
        launch.actions.DeclareLaunchArgument(
            name='cfg',
            default_value='cfg.yaml',
            description='file name of experiment configuration yaml'
        ),
        launch.actions.DeclareLaunchArgument(
            name='render',
            default_value=str(True),
            description='launch the Ignition environment in a rendering window'
        ),
        launch.actions.OpaqueFunction(function=opaque_func)
    ])