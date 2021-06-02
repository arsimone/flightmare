#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import time
import math
import argparse
import numpy as np
import torch  
from shutil import copytree
from shutil import copy2
from pathlib import Path

import numpy as np
import math
from typing import List
from scipy.spatial.transform import Rotation 
from inputimeout import inputimeout, TimeoutOccurred

from rpg_baselines.envs import vec_env_wrapper as wrapper
from flightgym import QuadrotorEnv_v1
from stable_baselines3.ppo.ppo import PPO

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    self._min_reference_velocity_magnitude = 1.0
    self._reference_velocity_magnitude = 2.5
    self._model = [] # need to initialize
    # initialization of queue and stuff
    # self._weights_name = "/root/challenge/flightmare/flightrl/examples/saved/snaga_training/mlp_velocity_tracker_best_model.zip"
    self._weights_name = os.environ["FLIGHTMARE_PATH"] + "/flightrl/examples/saved/snaga_training/mlp_velocity_tracker_best_model.zip"
    
    cfg_vec_env = YAML().load(open(os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs/vec_env.yaml", 'r'))
    cfg_vec_env["env"]["render"] = "yes"
    cfg_vec_env["env"]["num_envs"] = 1
    cfg_vec_env["env"]["num_threads"] = 1    
    self.ch = "g"
            
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg_vec_env, Dumper=RoundTripDumper), False))
    
    self._model = PPO.load(path = self._weights_name, env=env)
            
  def getActions(self, obs, done, images, current_goal_position):
    # 1) transform input and create queue
    model_input = self.processInput(obs, done, images, current_goal_position)
    print("################################################################")
    print("Tracking error : ", model_input[0,:3])
    print("Rot mat 1 : ", model_input[0,3:6])
    print("Rot mat 2 : ", model_input[0,6:9])
    print("Rot mat 3 : ", model_input[0,9:12])
    print("Drone Vel      : ", model_input[0,12:15])
    print("Drone ang vel  : ", model_input[0,15:18])
    
    velocity_reference_abs = self.r.apply(obs[0,:3]+obs[0,12:15])
    print("velocity_reference_abs  : ", velocity_reference_abs)
    
    if (self.ch == "g") or (self.ch == "p"):
      try:
        self.ch = inputimeout(prompt="", timeout=0.00001)
      except TimeoutOccurred:
        pass
    else:
      self.ch = input()
                
    # 2) get actions from model
    action, _ = self._model.predict(model_input, deterministic=True)

    return action
    
  def processInput(self, obs, done, images, current_goal_position):
    # 1) get velocity
    velocity_reference = self.calculateVelocity(obs, current_goal_position)
    print("velocity_reference  : ", velocity_reference)
    # 2) transform velocity in correct frame
    self.r = Rotation.from_matrix([ [obs[0, 3], obs[0, 6], obs[0,  9] ] ,
                               [obs[0, 4], obs[0, 7], obs[0, 10] ] ,
                               [obs[0, 5], obs[0, 8], obs[0, 11] ] ])    
    
    velocity_reference_body = self.r.inv().apply(velocity_reference)
    
    # 3) get tracking error
    tracking_error = velocity_reference_body - obs[0, 12:15]
    obs[0, :3] = tracking_error
    return obs
    
  def calculateVelocity(self, obs, current_goal_position):
    goal_relative_position = current_goal_position - obs[0, 0:3]
    goal_relative_position_norm = np.linalg.norm(goal_relative_position)
    
    velocity_reference_versor = goal_relative_position/goal_relative_position_norm
    velocity_reference = velocity_reference_versor * self.determine_velocity_magnitude(goal_relative_position_norm)
    
    return velocity_reference
  
  def determine_velocity_magnitude(self, goal_distance):
    scaling_factor = 2*math.atan((goal_distance)) / math.pi
    velocity_magnitude = scaling_factor * self._reference_velocity_magnitude
    if (velocity_magnitude < self._min_reference_velocity_magnitude):
      return self._min_reference_velocity_magnitude
    else:
      return velocity_magnitude 