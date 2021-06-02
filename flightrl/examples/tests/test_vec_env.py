#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import time
import numpy as np

from rpg_baselines.envs import tcn_mlp_ae_vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(
                dump(cfg, Dumper=RoundTripDumper), False))
    env.test = True

    action = np.zeros([env.num_envs, env.num_acts], dtype=np.float32)
    action += np.array([-0.01, -0.01, 0.00, 0.00])
        
    connectedToUnity = False 
    connectedToUnity = env.connectUnity()  
    # while not connectedToUnity:
    #     connectedToUnity = env.connectUnity()             
    #     if not connectedToUnity:  
    #         print("Couldn't connect to unity, will try another time.")    
    
    # print("env.num_envs : ", env.num_envs)

    max_ep_length = env.max_episode_steps

    if env.num_envs == 1:
        object_density_fractions = np.ones([env.num_envs], dtype=np.float32)
    else:
        object_density_fractions = np.linspace(0.0, 1.0, num=env.num_envs)

    # object_density_fractions = np.random.rand(env.num_envs)
    
    env.set_objects_densities(object_density_fractions=object_density_fractions)
    time.sleep(5)
    print(object_density_fractions)
    env.reset()



    # print("max_ep_length : ", max_ep_length)
    
    done, ep_len = False, 0
    
    while not ((ep_len >= max_ep_length*10)):
        
        index = 0
        observations, reward, done, infos = env.step(action)
        # print("RelVelCommand ", observations[index, :3])
        # print("RotMat 1 col: ", observations[index,3:6])
        # print("RotMat 2 col: ", observations[index, 6:9])
        # print("RotMat 3 col: ", observations[index, 9:12])
        # print("VelLin: ", observations[index, 12:15])
        # print("AngRates: ", observations[index, 0, 15:18, -1])
        # print("RelPosGoal: ", observations[1, 0, 18:21, -1])

        # print(reward)
        # print(ep_len)
        # print("###############")
        ep_len += 1


if __name__ == "__main__":
    main()