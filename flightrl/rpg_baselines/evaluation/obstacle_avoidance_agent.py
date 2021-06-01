#!/usr/bin/env python3

import numpy as np
import math
from typing import List

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
        
  def getActions(self, obs, done, images, current_goal_position):
    # 1) transform input and create queue
    model_input = self.processInput(obs, done, images, current_goal_position)
    # 2) get actions from model
    action, _ = self._model.predict(obs, deterministic=True)

    return action
    
  def processInput(self, obs, done, images, current_goal_position):
    # 1) get velocity
    velocity_reference = self.calculateVelocity( obs, current_goal_position)
    
    # 2) trasformation of obs into what policy needs
    
    # 3) ae
    
    # 4) update tcn queue 
    
    return tcn_queue
    
  def calculateVelocity(self, obs, current_goal_position):
    drone_position = obs[0:3]
    goal_relative_position = current_goal_position - drone_position
    goal_relative_position_norm = np.linalg.norm(goal_relative_position)
    
    velocity_reference_versor = goal_relative_position/goal_relative_position_norm
    velocity_reference = velocity_reference_versor * self.determine_velocity_magnitude(goal_relative_position_norm)
    
    return velocity_reference
  
  def determine_velocity_magnitude(self, goal_distance):
    scaling_factor = 2*math.atan((goal_distance - self.switch_goal_distance + 0.5)) / math.pi
    velocity_magnitude = scaling_factor * self._reference_velocity_magnitude
    if (velocity_magnitude < self._min_reference_velocity_magnitude):
      return self._min_reference_velocity_magnitude
    else:
      return velocity_magnitude 