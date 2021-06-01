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
    
    # initialization
        
  def getActions(self, obs, done, images, current_goal_position):
    action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    action[0,0] += -0.01
    return action
  
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