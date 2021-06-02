#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import time
from shutil import copytree
from shutil import copy2
from pathlib import Path

#
from stable_baselines3.common import logger

#
from stable_baselines3.ppo.ppo_custom import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
# from rpg_baselines.ppo.mlp_ae_ppo2_test import test_model
from test_flightmare.test_model_goal import test_model
from rpg_baselines.envs import tcn_mlp_ae_vec_env_wrapper as wrapper
import rpg_baselines.common.util as U
from rpg_baselines.models.tcn_mlp import CustomActorCriticPolicy
#
from flightgym import QuadrotorEnv_v1


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='/home/arsimone/master-thesis/flightmare/flightrl/examples/saved/snaga_training/tcn_velocity_tracker_45000000_steps.zip', #velocity_tracker_3000000_steps.zip',#velocity_tracker_50000000_steps.zip',
                        help='trained weight path')
    parser.add_argument('-f', '--file', type=str, default=None,
                        help='continue training from previous weights')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    cfg_rl = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightrl/configs/rl_parameters.yaml", 'r')) 
    cfg_env = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/quadrotor_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
    
    render_wrapper = cfg["env"]["render"] == "yes"

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    env.render = render_wrapper
    connectedToUnity = False 
    if args.render:
        while not connectedToUnity:        
            connectedToUnity = env.connectUnity()             
            if not connectedToUnity:  
                print("Couldn't connect to unity, will try another time.") 


    # set random seed
    configure_random_seed(args.seed, env=env)

    # set object density
    # object_density_fractions = np.ones(env.num_envs)
    object_density_fractions = np.linspace(0.3, 1.0, num=env.num_envs, dtype=np.float32)
    if args.render:
        env.set_objects_densities(object_density_fractions = object_density_fractions)
    time.sleep(5)
    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)

        # Copy config file in right folder.
        # Flightlib configs
        flightlib_configs_folder_input = os.environ["FLIGHTMARE_PATH"] + "/flightlib/configs"
        configs_folder_output = saver.data_dir + "/configs"
        copytree(flightlib_configs_folder_input, configs_folder_output)
        # Flightrl config file
        flightrl_config_file_input = os.environ["FLIGHTMARE_PATH"] + "/flightrl/configs/rl_parameters.yaml"
        copy2(flightrl_config_file_input, configs_folder_output)
        print("Config files copied in ", configs_folder_output)

        policy_kwargs = dict(
            features_extractor_class=None,
        )
        loader_kwargs = dict(
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log=saver.data_dir,
            n_steps = 400,
        )

        def linear_lr(progress_remaining):
            return cfg_rl['schedule_lr_start_value']*progress_remaining + (1-progress_remaining)*cfg_rl['schedule_lr_end_value']

        checkpoint_model_cb = CheckpointCallback(save_freq = int(cfg_rl['checkpoint_interval'] / (env.num_envs)), save_path = saver.data_dir, name_prefix='tcn_velocity_tracker')

        if args.file:
            model = PPO.load(args.file, env, reward_baseline=cfg_rl['reward_baseline'])
            model.tensorboard_log = saver.data_dir

        else:
            model = PPO(
                tensorboard_log=saver.data_dir,
                policy=CustomActorCriticPolicy,  # check activation function
                policy_kwargs=dict(policy_dim=cfg_rl['net_arch_pi'],
                                   value_function_dim=cfg_rl['net_arch_vf'],
                                   tcn_dim=cfg_rl['tcn_net_arch'],
                                   tcn_buffer_size=env.tcn_input_size),
                env=env,
                gae_lambda=cfg_rl['gae_lambda'],
                gamma=cfg_rl['gamma'],  # lower 0.9 ~ 0.99
                n_steps=int(cfg_rl['n_steps']/env.num_envs),
                ent_coef=cfg_rl['ent_coef'],
                learning_rate=linear_lr,
                vf_coef=cfg_rl['vf_coef'],
                max_grad_norm=cfg_rl['max_grad_norm'],
                batch_size=cfg_rl['n_steps'],
                n_epochs=cfg_rl['n_epochs'],
                clip_range=cfg_rl['clip_range'],
                verbose=2,
            )


        # tensorboard
        # Make sure that your chrome browser is already on.

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)

        model.learn(
            total_timesteps=int(100000000),
            callback=checkpoint_model_cb,
            eval_log_path=saver.data_dir)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        print(args.weight)
        env.test = True
        model = PPO.load(args.weight, env)
        test_model(env, model, args.weight, cfg_env, cfg_rl)
        # env._out.release()


if __name__ == "__main__":
    main()
