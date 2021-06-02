import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import datetime
import os
import sys
from inputimeout import inputimeout, TimeoutOccurred




from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.buffers_custom import RolloutBuffer as RolloutBufferCustom
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.max_ep_rew_mean = 700.0

        self.rollout_collection_time_ = 0
        self.net_train_time_ = 0

        if _init_setup_model:
            self._setup_model()

        self.rewards_components_envs_cumulative = np.zeros([env.num_envs, 7], dtype=float)
        self.rewards_components_envs_sum = np.zeros([1, 7], dtype=float)
        self.terminal_rewards_sum = 0
        self.terminal_causes = np.zeros([5], dtype=float)
        self.terminated_envs = 0

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.prediction_steps = 5 # how far in the future is the net require to predict in term of steps (0.2s per step)
        # self.decoding_history = th.zeros((self.n_steps - self.prediction_steps, self.n_envs, self.observation_space.shape[0])).to(self.device)
        # self.observation_history = th.zeros((self.n_steps - self.prediction_steps, self.n_envs, self.observation_space.shape[0])).to(self.device)
        # self.decoding_mask = th.ones((self.n_steps - self.prediction_steps, self.n_envs)).to(self.device)

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.rewards_components_envs_sum = np.zeros([1, 7], dtype=float)
        self.terminal_rewards_sum = 0
        self.terminal_causes = np.zeros([5], dtype=float)
        self.terminated_envs = 0

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            # if n_steps < n_rollout_steps - self.prediction_steps:
            #     self.decoding_history[n_steps, :, :] = decoding
            # if n_steps > self.prediction_steps:
            #     self.observation_history[n_steps - self.prediction_steps, :, :] = obs_tensor[:,:,0]

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if(self.num_timesteps < 400):
                print("After step ", n_steps)

            # if(self.num_timesteps % 50000 == 0):
            #     env.set_objects_densities()
            
            # Deal with extra_info
            for env_idx, info in enumerate(infos):
                env_extra_info = info.get("extra_info")
                self.rewards_components_envs_cumulative[env_idx, 0] += env_extra_info['lin_vel_ori_rew']
                self.rewards_components_envs_cumulative[env_idx, 1] += env_extra_info['lin_vel_mag_rew']
                self.rewards_components_envs_cumulative[env_idx, 2] += env_extra_info['ori_rew']
                self.rewards_components_envs_cumulative[env_idx, 3] += env_extra_info['ang_vel_rew']
                self.rewards_components_envs_cumulative[env_idx, 4] += env_extra_info['act_rew']
                self.rewards_components_envs_cumulative[env_idx, 5] += env_extra_info['survival_rew']
                self.rewards_components_envs_cumulative[env_idx, 6] += 0
                if dones[env_idx]:
                    self.terminal_rewards_sum += env_extra_info['term_rew']
                    self.terminal_causes[int(env_extra_info['term_reason']-1)] += 1
                    self.terminated_envs += 1
                    self.rewards_components_envs_sum += self.rewards_components_envs_cumulative[env_idx, :]
                    self.rewards_components_envs_cumulative[env_idx, :] = np.zeros([1, 7], dtype=float)
                    # self.decoding_mask[max(n_steps-self.prediction_steps, 0):n_steps, env_idx] = 0
                    

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.generate_mask()

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        graph = logger.Graph(self.policy, th.as_tensor(self._last_obs).to(self.device)) 
        logger.record("graph", graph)

        while self.num_timesteps < total_timesteps:
            
            time_before_collect_rollout = time.time()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            time_after_collect_rollout = time.time()

            logger.record("time/rollout_collection", (time_after_collect_rollout - time_before_collect_rollout))

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("main/ep_mean_total_reward", ep_rew_mean)
                    logger.record("main/ep_mean_length", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                if self.terminated_envs != 0:
                    self.rewards_components_envs_sum = self.rewards_components_envs_sum / self.terminated_envs
                    logger.record("reward_components/ep_mean_lin_vel_ori_rew", self.rewards_components_envs_sum[0,0]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_lin_vel_mag_rew", self.rewards_components_envs_sum[0,1]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_ori_rew", self.rewards_components_envs_sum[0,2]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_ang_vel_rew", self.rewards_components_envs_sum[0,3]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_act_rew", self.rewards_components_envs_sum[0,4]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_distances_rew", self.rewards_components_envs_sum[0,6]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_survival_rew", self.rewards_components_envs_sum[0,5])
                    logger.record("reward_components/ep_mean_terminal_rew", (self.terminal_rewards_sum / self.terminated_envs))
                    
                    self.terminal_causes = self.terminal_causes / self.terminated_envs
                    logger.record("termination_info/term_crash_world_box", self.terminal_causes[0])
                    logger.record("termination_info/term_collision", self.terminal_causes[1])
                    logger.record("termination_info/term_time_is_up", self.terminal_causes[2])
                    logger.record("termination_info/term_velocity_too_off", self.terminal_causes[3])

                logger.record("main/fps", fps)
                logger.record("main/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("main/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)
                
                # Save model if best reward so far. Start saving the model after reward is above threshold.
                if (ep_rew_mean > self.max_ep_rew_mean):
                    self.max_ep_rew_mean = ep_rew_mean
                    path = os.path.join(eval_log_path, "best_rl_model")
                    self.save(path)
                    print("Saving best model with reward : ", self.max_ep_rew_mean)
                else:
                    print("Best model has reward : ", self.max_ep_rew_mean)
                
                # Save model on request.
                try:
                    ch = inputimeout(prompt="Useful info: ", timeout=0.01)
                except TimeoutOccurred:
                    ch = ""
                if ch == "s":
                    file_name = "rl_model_" + str(self.num_timesteps) + "_steps" 
                    path = os.path.join(eval_log_path, file_name)
                    self.save(path)
                    print("Saving model on request.")
                            
                # Print info regarding time remaining to training end based on current fps.
                steps_remaining = total_timesteps - self.num_timesteps
                if steps_remaining != 0:
                    estimated_time = str(datetime.timedelta(seconds=steps_remaining/fps))
                    print("Remaining steps : ", steps_remaining, "\nEstimated time of completion : ", estimated_time);         
                else:
                    print("Training completed!")
            
            time_after_logging = time.time()
            logger.record("time/logging_time", time_after_logging - time_after_collect_rollout)
            print("Logging time : %s seconds" % (time_after_logging - time_after_collect_rollout))
            
            self.train()

            self.net_train_time_ = time.time() - time_after_collect_rollout
            print("Collect rollout time : %s seconds" % (time_after_collect_rollout - time_before_collect_rollout))
            logger.record("time/net_train_time", time.time() - time_after_logging)
            print("Go through data time : %s seconds" % (time.time() - time_after_logging))

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []


class OnPolicyAlgorithmCustom(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(OnPolicyAlgorithmCustom, self).__init__(
            policy=policy,
            env=env,
            policy_base=ActorCriticPolicy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.max_ep_rew_mean = 700.0

        self.rollout_collection_time_ = 0
        self.net_train_time_ = 0

        if _init_setup_model:
            self._setup_model()

        self.rewards_components_envs_cumulative = np.zeros([env.num_envs, 7], dtype=float)
        self.rewards_components_envs_sum = np.zeros([1, 7], dtype=float)
        self.terminal_rewards_sum = 0
        self.terminal_causes = np.zeros([5], dtype=float)
        self.terminated_envs = 0

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBufferCustom(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.prediction_steps = 5 # how far in the future is the net require to predict in term of steps (0.2s per step)
        # self.decoding_history = th.zeros((self.n_steps - self.prediction_steps, self.n_envs, self.observation_space.shape[0])).to(self.device)
        # self.observation_history = th.zeros((self.n_steps - self.prediction_steps, self.n_envs, self.observation_space.shape[0])).to(self.device)
        # self.decoding_mask = th.ones((self.n_steps - self.prediction_steps, self.n_envs)).to(self.device)

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBufferCustom, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.rewards_components_envs_sum = np.zeros([1, 7], dtype=float)
        self.terminal_rewards_sum = 0
        self.terminal_causes = np.zeros([5], dtype=float)
        self.terminated_envs = 0

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            # if n_steps < n_rollout_steps - self.prediction_steps:
            #     self.decoding_history[n_steps, :, :] = decoding
            # if n_steps > self.prediction_steps:
            #     self.observation_history[n_steps - self.prediction_steps, :, :] = obs_tensor[:,:,0]

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            if(self.num_timesteps < 400):
                print("After step ", n_steps)

            # if(self.num_timesteps % 50000 == 0):
            #     env.set_objects_densities()
            
            # Deal with extra_info
            for env_idx, info in enumerate(infos):
                env_extra_info = info.get("extra_info")
                self.rewards_components_envs_cumulative[env_idx, 0] += env_extra_info['lin_vel_ori_rew']
                self.rewards_components_envs_cumulative[env_idx, 1] += env_extra_info['lin_vel_mag_rew']
                self.rewards_components_envs_cumulative[env_idx, 2] += env_extra_info['ori_rew']
                self.rewards_components_envs_cumulative[env_idx, 3] += env_extra_info['ang_vel_rew']
                self.rewards_components_envs_cumulative[env_idx, 4] += env_extra_info['act_rew']
                self.rewards_components_envs_cumulative[env_idx, 5] += env_extra_info['survival_rew']
                self.rewards_components_envs_cumulative[env_idx, 6] += 0
                if dones[env_idx]:
                    self.terminal_rewards_sum += env_extra_info['term_rew']
                    self.terminal_causes[int(env_extra_info['term_reason']-1)] += 1
                    self.terminated_envs += 1
                    self.rewards_components_envs_sum += self.rewards_components_envs_cumulative[env_idx, :]
                    self.rewards_components_envs_cumulative[env_idx, :] = np.zeros([1, 7], dtype=float)
                    # self.decoding_mask[max(n_steps-self.prediction_steps, 0):n_steps, env_idx] = 0
                    

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.generate_mask()

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        graph = logger.Graph(self.policy, th.as_tensor(self._last_obs).to(self.device)) 
        logger.record("graph", graph)

        while self.num_timesteps < total_timesteps:
            
            time_before_collect_rollout = time.time()
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            time_after_collect_rollout = time.time()

            logger.record("time/rollout_collection", (time_after_collect_rollout - time_before_collect_rollout))

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("main/ep_mean_total_reward", ep_rew_mean)
                    logger.record("main/ep_mean_length", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                if self.terminated_envs != 0:
                    self.rewards_components_envs_sum = self.rewards_components_envs_sum / self.terminated_envs
                    logger.record("reward_components/ep_mean_lin_vel_ori_rew", self.rewards_components_envs_sum[0,0]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_lin_vel_mag_rew", self.rewards_components_envs_sum[0,1]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_ori_rew", self.rewards_components_envs_sum[0,2]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_ang_vel_rew", self.rewards_components_envs_sum[0,3]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_act_rew", self.rewards_components_envs_sum[0,4]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_distances_rew", self.rewards_components_envs_sum[0,6]/safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]) )
                    logger.record("reward_components/ep_mean_survival_rew", self.rewards_components_envs_sum[0,5])
                    logger.record("reward_components/ep_mean_terminal_rew", (self.terminal_rewards_sum / self.terminated_envs))
                    
                    self.terminal_causes = self.terminal_causes / self.terminated_envs
                    logger.record("termination_info/term_crash_world_box", self.terminal_causes[0])
                    logger.record("termination_info/term_collision", self.terminal_causes[1])
                    logger.record("termination_info/term_time_is_up", self.terminal_causes[2])
                    logger.record("termination_info/term_velocity_too_off", self.terminal_causes[3])

                logger.record("main/fps", fps)
                logger.record("main/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                logger.record("main/total_timesteps", self.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.num_timesteps)
                
                # Save model if best reward so far. Start saving the model after reward is above threshold.
                if (ep_rew_mean > self.max_ep_rew_mean):
                    self.max_ep_rew_mean = ep_rew_mean
                    path = os.path.join(eval_log_path, "best_rl_model")
                    self.save(path)
                    print("Saving best model with reward : ", self.max_ep_rew_mean)
                else:
                    print("Best model has reward : ", self.max_ep_rew_mean)
                
                # Save model on request.
                try:
                    ch = inputimeout(prompt="Useful info: ", timeout=0.01)
                except TimeoutOccurred:
                    ch = ""
                if ch == "s":
                    file_name = "rl_model_" + str(self.num_timesteps) + "_steps" 
                    path = os.path.join(eval_log_path, file_name)
                    self.save(path)
                    print("Saving model on request.")
                            
                # Print info regarding time remaining to training end based on current fps.
                steps_remaining = total_timesteps - self.num_timesteps
                if steps_remaining != 0:
                    estimated_time = str(datetime.timedelta(seconds=steps_remaining/fps))
                    print("Remaining steps : ", steps_remaining, "\nEstimated time of completion : ", estimated_time);         
                else:
                    print("Training completed!")
            
            time_after_logging = time.time()
            logger.record("time/logging_time", time_after_logging - time_after_collect_rollout)
            print("Logging time : %s seconds" % (time_after_logging - time_after_collect_rollout))
            
            self.train()

            self.net_train_time_ = time.time() - time_after_collect_rollout
            print("Collect rollout time : %s seconds" % (time_after_collect_rollout - time_before_collect_rollout))
            logger.record("time/net_train_time", time.time() - time_after_logging)
            print("Go through data time : %s seconds" % (time.time() - time_after_logging))

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
