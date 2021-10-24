# Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import math
import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional, Tuple, TypedDict, Union, List, Iterable, Type, Callable, Dict
VecEnvIndices = Union[None, int, Iterable[int]]

import rclpy
from rclpy.node import Node
from rclpy.client import Client, Future
from gym_ignition.srv import Step, Reset, Info, Observe, Extra

class Clients(TypedDict):
    srv: List[Client]

class GymIgnitionVecEnv(Node, VecEnv):
    def __init__(self, cfg, clip_obs: np.float_ = 10.0):
        super().__init__(node_name="vecenv")

        self._max_episode_steps = math.floor(cfg['max_time'] / cfg['control_dt'])
        self._num_envs: int = cfg['num_envs']

        # Create the service clients
        self._clients: Clients = {"step" : [], "reset" : [], "info" : [], "observe" : [], "extra" : []}
        [[self._clients[j].append(self.create_client(k, '/env' + str(i) + '/{}'.format(j))) for j, k in zip(["step", "reset", "info", "observe", "extra"], [Step, Reset, Info, Observe, Extra])] for i in range(self.num_envs)]

        msg: Info.Request = Info.Request()
        msg.sim_time_step = cfg["step_size"]
        msg.control_time_step = cfg["control_dt"]
        futures = []
        # Set the simulation and control time steps for each environment
        [futures.append(self._clients["info"][i].call_async(msg)) for i in range(self.num_envs)]
        self._check_futures_status(futures, self._info_callback)
        
        self._observation_space: spaces.Box = spaces.Box(-np.ones((self.num_envs, self.num_obs)), np.ones((self.num_envs, self.num_obs)))
        self._action_space: spaces.Box = spaces.Box(-np.ones((self.num_envs, self.num_acts)), np.ones((self.num_envs, self.num_acts)), dtype=np.float32)

        self._observation: np.ndarray = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._extraInfo: Dict[str, np.ndarray] = dict()
        self._rewards: Dict[str, np.ndarray] = dict()
        
        self._reward: np.ndarray = np.zeros(self.num_envs, dtype=np.float32)
        self._done: np.ndarray = np.zeros((self.num_envs), dtype=np.bool)

        if cfg['camera']['has_rgb']:
            self._rgb_cam_observation: np.ndarray = np.zeros([
                self.num_envs,
                cfg['camera']['rgb']['resolution']['w'],
                cfg['camera']['rgb']['resolution']['h'],
                3]).astype(np.uint8)
            self._rgb_cam_extra: np.ndarray = np.zeros([
                self.num_envs,
                cfg['camera']['rgb']['resolution']['w'],
                cfg['camera']['rgb']['resolution']['h'],
                3]).astype(np.uint8)
        else:
            self._rgb_cam_observation: np.ndarray = np.zeros([
                self.num_envs, 1, 1, 3]).astype(np.uint8)
            self._rgb_cam_extra: np.ndarray = np.zeros([
                self.num_envs, 1, 1, 3]).astype(np.uint8)

        if cfg['camera']['has_depth']:
            self._depth_cam_observation: np.ndarray = np.zeros([
                self.num_envs,
                cfg['camera']['depth']['resolution']['w'],
                cfg['camera']['depth']['resolution']['h']]).astype(np.float32)
            self._depth_cam_extra: np.ndarray = np.zeros([
                self.num_envs,
                cfg['camera']['depth']['resolution']['w'],
                cfg['camera']['depth']['resolution']['h']]).astype(np.float32)
        else:
            self._depth_cam_observation: np.ndarray = np.zeros([
                self.num_envs, 1, 1]).astype(np.float32)
            self._depth_cam_extra: np.ndarray = np.zeros([
                self.num_envs, 1, 1]).astype(np.float32)

        if cfg['camera']['has_thermal']:
            self._thermal_cam_observation: np.ndarray = np.zeros([
                self.num_envs,
                cfg['camera']['thermal']['resolution']['w'],
                cfg['camera']['thermal']['resolution']['h']]).astype(np.uint16)
            self._thermal_cam_extra: np.ndarray = np.zeros([
                self.num_envs,
                cfg['camera']['thermal']['resolution']['w'],
                cfg['camera']['thermal']['resolution']['h']]).astype(np.uint16)
        else:
            self._thermal_cam_observation: np.ndarray = np.zeros([
                self.num_envs, 1, 1]).astype(np.uint16)
            self._thermal_cam_extra: np.ndarray = np.zeros([
                self.num_envs, 1, 1]).astype(np.uint16)

        self._clip_obs: np.float_ = clip_obs
        self.dynamics_randomization: bool = False # TODO: enable randomizing the environment through this class

        self._seed = 0
    # TODO: set the environment random seed
    # def seed(self, seed=None):
    #     self.wrapper.setSeed(seed)

    def _populate_camera_observation(self, response: Union[Step.Response, Reset.Response, Extra.Response, Observe.Response]):

        # Get the latest camera images
        if response.rgb.data:
            self._rgb_cam_observation[response.env_id, :, :, :] = np.reshape(response.rgb.data, (self._rgb_cam_observation.shape[1], self._rgb_cam_observation.shape[2], self._rgb_cam_observation.shape[3]))

        if response.depth.data:
            self._depth_cam_observation[response.env_id, :, :] = np.reshape(response.depth.data, (self._depth_cam_observation.shape[1], self._depth_cam_observation.shape[2]))

        if response.thermal.data:
            self._thermal_cam_observation[response.env_id, :, :] = np.reshape(response.thermal.data, (self._thermal_cam_observation.shape[1], self._thermal_cam_observation.shape[2]))

    def _info_callback(self, response: Info.Response):

        self._num_obs: int = response.num_obs
        self._num_acts: int = response.num_acts
        self._num_extras: int = response.num_extras
        if response.env_id == 0:
            self._visualizable: bool = response.visualizable

    def _observe_callback(self, response: Observe.Response):

        self._observation[response.env_id, :] = response.observation

        self._populate_camera_observation(response)

    def _reset_callback(self, response: Reset.Response):

        self._observation[response.env_id, :] = response.observation

        # Get the latest camera images
        self._populate_camera_observation(response)

    def _step_callback(self, response: Step.Response):

        self._observation[response.env_id, :] = response.observation
        self._reward[response.env_id] = response.reward
        self._done[response.env_id] = response.is_done
        
        # Get the latest camera images
        self._populate_camera_observation(response)

    def _extra_callback(self, response: Extra.Response):
        
        if response.extra_info_keys:
            for idx, key in enumerate(response.extra_info_keys):
                if not key in self._extraInfo:
                    self._extraInfo[key] = np.zeros((self.num_envs,), dtype=np.float32)
                else:
                    self._extraInfo[key][response.env_id] = response.extra_info[idx]

        if response.reward_keys:
            for idx, key in enumerate(response.reward_keys):
                if not key in self._rewards:
                    self._rewards[key] = np.zeros((self.num_envs,), dtype=np.float32)
                else:
                    self._rewards[key][response.env_id] = response.reward[idx]

        if response.rgb.data:
            self._rgb_cam_extra[response.env_id, :, :, :] = np.reshape(response.rgb.data, (self._rgb_cam_extra.shape[1], self._rgb_cam_extra.shape[2], self._rgb_cam_extra.shape[3]))

        if response.depth.data:
            self._depth_cam_extra[response.env_id, :, :] = np.reshape(response.depth.data, (self._depth_cam_extra.shape[1], self._depth_cam_extra.shape[2]))

        if response.thermal.data:
            self._thermal_cam_extra[response.env_id, :, :] = np.reshape(response.thermal.data, (self._thermal_cam_extra.shape[1], self._thermal_cam_extra.shape[2]))

    def _check_futures_status(self, futures: List[Future], callback: Callable):

        response_received: List[bool] = [False] * self.num_envs
        while not all(response_received):
            rclpy.spin_once(self)
            for i in range(len(futures)):
                if futures[i].done() and not response_received[i]:
                    try:
                        response = futures[i].result()
                    except Exception as e:
                        self.get_logger().info('[GymIgnitionVecEnv.py] Service call failed for environment {}. Exception: {}'.format(str(i), e))
                        raise RuntimeError('[GymIgnitionVecEnv.py] Service call failed for environment {}.'.format(str(i))) from e
                    
                    callback(response)
                    response_received[i] = True

    def step(self, action, visualize=False):

        msg: Step.Request = Step.Request()
        futures: List[Future] = []
        # Send the action to each environment
        def set_action(i):
            msg.action = [np.float_(a) for a in action[i, :]]
            return msg
        [futures.append(self._clients["step"][i].call_async(set_action(i))) for i in range(self.num_envs)]
        self._check_futures_status(futures, self._step_callback)

        self.extras()
        return self._observation.copy(), self._reward.copy(), self._done.copy(), self.get_extras()

    def observe(self):

        msg: Observe.Request = Observe.Request()
        futures: List[Future] = []
        # Get the latest observation from the environment
        [futures.append(self._clients["observe"][i].call_async(msg)) for i in range(self.num_envs)]
        self._check_futures_status(futures, self._observe_callback)

        return self._observation.copy()

    def get_extras(self):

        return self._extraInfo.copy()

    def get_rgb_observation(self):

        return self._rgb_cam_observation.copy()

    def get_depth_observation(self):

        return self._depth_cam_observation.copy()

    def get_thermal_observation(self):

        return self._thermal_cam_observation.copy()

    def reset(self):
        self._reward: np.ndarray = np.zeros(self.num_envs, dtype=np.float32)
        self._done: np.ndarray = np.zeros(self.num_envs, dtype=np.bool)

        msg: Reset.Request = Reset.Request()
        futures: List[Future] = []
        # Reset each environment
        [futures.append(self._clients["reset"][i].call_async(msg)) for i in range(self.num_envs)]
        self._check_futures_status(futures, self._reset_callback)

        return self._observation.copy()

    def extras(self):

        msg: Extra.Request = Extra.Request()
        futures: List[Future] = []
        # Query the extra information from the environment (dictionary of values specified by the user, extra RGB, depth, and thermal image, and the individual reward terms)
        [futures.append(self._clients["extra"][i].call_async(msg)) for i in range(self.num_envs)]
        self._check_futures_status(futures, self._extra_callback)

        return self._extraInfo.copy(), self._rewards.copy(), self._rgb_cam_extra.copy(), self._depth_cam_extra.copy(), self._thermal_cam_extra.copy()

    def render(self, mode='human') -> bool:
        return self._visualizable

    def close(self):

        self.destroy_node()
        rclpy.shutdown()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self._seed = seed

        return [None]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        raise RuntimeError('This method is not implemented')

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name: str, indices: List[int] = None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name: str, value, indices: List[int] = None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name: str, *method_args: Tuple, indices: List[int] = None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    def print(self, msg: str):

        self.get_logger().info(msg)

    @property
    def metadata(self) -> None:
        return None

    @property
    def reward_range(self) -> List[float]:
        return [-float('inf'), float('inf')]

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_obs(self) -> int:
        return self._num_obs

    @property
    def num_acts(self) -> int:
        return self._num_acts

    @property
    def num_extras(self) -> int:
        return self._num_extras

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def is_metaworld(self) -> bool:
        return False

    @property
    def max_timesteps(self) -> float:
        return self._max_episode_steps

    @property
    def is_async(self) -> bool:
        return False # return false because all the environments are rolled out in parallel (synchronized)

    def __len__(self):
        return self._num_envs