
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv
import cv2 as cv
import torch

import sys
import os

sys.path.insert(1, os.environ['FLIGHTMARE_AE_PATH'])

from models.compressive_ae_8_8_8 import CAE


class FlightEnvVec(VecEnv):
    #
    def __init__(self, impl):
        self.wrapper = impl
        self.pub_port = 10253
        self.sub_port = 10254
        self.test = False   # Flag for testing time
        self.render = True   # Flag for testing time
        self.count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.ae_fts = 512  # number of autoencoder features
        self.ae_weight = os.environ['FLIGHTMARE_AE_PATH'] + '/saved/CAE_512_CP_epoch1140.pth'
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActDim()
        self.frame_dim = self.wrapper.getFrameDim()
        print("Observations: ", self.num_obs)
        print("Actions: ", self.num_acts)
        print("image shape:", self.frame_dim)
        self._observation_space = spaces.Box(
            np.ones(self.num_obs + self.ae_fts) * -np.Inf,
            np.ones(self.num_obs + self.ae_fts) * np.Inf,
            dtype=np.float32)
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        self._observations = np.zeros((self.num_envs, self.num_obs + self.ae_fts), dtype=np.float32)
        self._odometry = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.imgs_array = np.zeros((self.num_envs, self.frame_dim[0]*self.frame_dim[1]), dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]

        self.net = CAE()
        self.net.load_state_dict(torch.load(self.ae_weight, map_location=self.device))
        self.net.to(self.device)
        for parameter in self.net.parameters():
            parameter.requires_grad = False
        self.max_episode_steps = 1500
        self.images = []

        # self._video_name = '~/Videos/tesi/onboard.avi'
        # self._fourcc = cv.VideoWriter_fourcc(*'MJPG')
        # self._out = cv.VideoWriter(self._video_name, self._fourcc, 20.0, (512, 1024))

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def set_objects_densities(self, object_density_fractions):
        if(self.render):
            self.wrapper.setObjectsDensities(object_density_fractions)
            print("density set to ", object_density_fractions)
        return

    def step(self, action):
        self.wrapper.step(action, self._odometry, self.imgs_array,
                          self._reward, self._done, self._extraInfo)
        self._observations[:, :self.num_obs] = self._odometry
        
        # ----- Uncomment below to check if images are correct -----
        # images = 255 - self.imgs_array.reshape((self.num_envs, self.frame_dim[0],
        #                                             self.frame_dim[1], 3), order='F')
        # cv.imshow('prova', images[0,:,:,0])
        # cv.waitKey(1)

        # ----- Uncomment below to encode images using AE -----
        if self.render:
            with torch.no_grad():
                images = 1 - self.imgs_array.reshape((self.num_envs, self.frame_dim[0],
                                                    self.frame_dim[1]), order='F')*15/6
                images[images<0] = 0
                self.images = images
                images_tensor = torch.from_numpy(images).float().to(self.device)
                
                # encode image
                encoded_rec, encoded = self.net.encode((images_tensor.unsqueeze(1)))
                self._observations[:, self.num_obs:] = encoded.cpu().detach().numpy().reshape(self.num_envs, self.ae_fts)
                
                if(self.test):
                    mask_tensor = self.net.decode(encoded_rec)
                    mask = (mask_tensor.cpu().detach().numpy().reshape((self.num_envs, self.frame_dim[0], self.frame_dim[1]), order='F')).astype(np.float32)
                    img_mask = cv.resize(np.concatenate((1-images[0,:,:], 1-mask[0,:,:]), axis=1), None, fx=4, fy=4,
                                        interpolation=cv.INTER_CUBIC)
                    cv.imshow('image - generated mask', img_mask)
                    # self._out.write(img_mask)
                    cv.waitKey(1)

        # ----- Uncomment below to check if images are correct on snaga-----
        # if self.count < 100:
        #     cv.imwrite('images/img'+str(self.count)+'.png', images[0,:,:,:])
        #     self.count = self.count + 1

        if len(self._extraInfoNames) is not 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0,
                    len(self._extraInfoNames))}} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()
        # print("[VecEnvWrapper/step] The observation shape is ",self._observation.shape)

        # env_extra_info = info[0].get("extra_info")
        # drone_position = np.asarray([env_extra_info['relative_pos_x'], env_extra_info['relative_pos_y'], env_extra_info['relative_pos_z']], dtype=np.float32)
        # self._observations[0, :3] = drone_position
        
        return self._observations.copy(), self._reward.copy(), \
            self._done.copy(), info.copy()
            
    def get_images(self):
        return self.images.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(action, self._odometry, self.imgs_array,
                                            self._reward, self._done, self._extraInfo, send_id)

        return receive_id

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._odometry, self.imgs_array)

        self._observations[:, :self.num_obs] = self._odometry
        images = self.imgs_array.reshape((self.num_envs, self.frame_dim[0],
                                                    self.frame_dim[1]), order='F')
        images_tensor = torch.from_numpy(images).float().to(self.device)
        # # convert BGR to RGB                                 
        # images_tensor = images_tensor[:, :, :, [2,1,0]]
        # encode image
        _, encoded = self.net.encode(images_tensor.unsqueeze(1))
        self._observations[:, self.num_obs:] = encoded.cpu().detach().numpy().reshape(self.num_envs, self.ae_fts)

        return self._observations.copy()

    def set_velocity_references(self, velocity_references):
        self.wrapper.setVelocityReferences(velocity_references)
        return

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        return self.wrapper.connectUnity(self.pub_port, self.sub_port)


    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_is_wrapped(self):
        """
        Check if environments are wrapped with a given wrapper.
        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        return np.ones([self.num_envs], dtype=bool).tolist()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')
