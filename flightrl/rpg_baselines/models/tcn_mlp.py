from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rpg_baselines.models.tcn import TemporalConvNet


class TCN_mlp(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        input_size: int,
        tcn_buffer_size: int = 32,
        tcn: List[int] = [256, 128, 128, 64],
        pi: List[int] = [256, 128, 64],
        vf: List[int] = [256, 128, 64],
        decoder: List[int] = [64, 128, 256],
    ):
        super(TCN_mlp, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = pi[-1]
        self.latent_dim_vf = vf[-1]
        layers_pi = []
        layers_vf = []
        layers_decoder_present = []
        layers_decoder_future1 = []
        layers_decoder_future2 = []
        
        # TCN
        self.tcn = TemporalConvNet(int(input_size/tcn_buffer_size), tcn)

        # Policy net
        for i in range(len(pi)):
            in_channels = tcn[-1]+18 if i == 0 else pi[i-1]
            out_channels = pi[i]
            layers_pi += [nn.Linear(in_channels, out_channels)]
            layers_pi += [nn.Tanh()]
        self.policy_net = nn.Sequential(*layers_pi)

        # Value function net
        for i in range(len(vf)):
            in_channels = tcn[-1]+18 if i == 0 else vf[i-1]
            out_channels = vf[i]
            layers_vf += [nn.Linear(in_channels, out_channels)]
            layers_vf += [nn.Tanh()]
        self.value_net = nn.Sequential(*layers_vf)

        # Decoder net for reconstructing present state
        for i in range(len(decoder)):
            in_channels = tcn[-1] if i == 0 else decoder[i-1]
            out_channels = decoder[i]
            layers_decoder_present += [nn.Linear(in_channels, out_channels)]
            layers_decoder_present += [nn.Tanh()]
        # last layer with the size of the input in order to recunstruct it 
        layers_decoder_present += [nn.Linear(decoder[-1], int(input_size/tcn_buffer_size)-18)]
        layers_decoder_present += [nn.Tanh()]
        self.decoder_net_present = nn.Sequential(*layers_decoder_present)

        # Decoder net for reconstructing future state distant 0.1s
        for i in range(len(decoder)):
            in_channels = tcn[-1] if i == 0 else decoder[i-1]
            out_channels = decoder[i]
            layers_decoder_future1 += [nn.Linear(in_channels, out_channels)]
            layers_decoder_future1 += [nn.Tanh()]
        # last layer with the size of the input in order to recunstruct it 
        layers_decoder_future1 += [nn.Linear(decoder[-1], int(input_size/tcn_buffer_size)-18)]
        layers_decoder_future1 += [nn.Tanh()]
        self.decoder_net_future1 = nn.Sequential(*layers_decoder_future1)

        # Decoder net for reconstructing future state distant 0.2s
        for i in range(len(decoder)):
            in_channels = tcn[-1] if i == 0 else decoder[i-1]
            out_channels = decoder[i]
            layers_decoder_future2 += [nn.Linear(in_channels, out_channels)]
            layers_decoder_future2 += [nn.Tanh()]
        # last layer with the size of the input in order to recunstruct it 
        layers_decoder_future2 += [nn.Linear(decoder[-1], int(input_size/tcn_buffer_size)-18)]
        layers_decoder_future2 += [nn.Tanh()]
        self.decoder_net_future2 = nn.Sequential(*layers_decoder_future2)

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        with th.no_grad():
            self.tcn_out = self.tcn(observations)
            # print(self.tcn_out[:,:,0])
            self.mlp_in = th.cat((self.tcn_out[:,:,0].detach(), observations[:,:18,0].detach()),1)
        
        return self.policy_net(self.mlp_in), self.value_net(self.mlp_in)

    def decode(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: th.Tensor decoding produced starting from tcn output
        """
        tcn_out = self.tcn(observations)
        # print(observations[:,18:,0])
        # mask_upscaled = cv.resize(1-observations[-1, 18:,0].cpu().detach().numpy().reshape(21,21).astype(np.float32), (512,512))
        # cv.imshow('image', mask_upscaled)
        # cv.waitKey(0)
        return tcn_out[:,:,0], self.decoder_net_present(tcn_out[:,:,0]), self.decoder_net_future1(tcn_out[:,:,0]), self.decoder_net_future2(tcn_out[:,:,0])


class Identity(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(Identity, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = Identity,
        policy_dim: List[int] = [256, 128, 64],
        value_function_dim: List[int] = [256, 128, 64],
        tcn_dim: List[int] = [256, 128, 128, 64],
        decoder_dim: List[int] = [256, 512, 512],
        tcn_buffer_size: int = 32,
        *args,
        **kwargs,
    ):
        self.policy_dim = policy_dim
        self.value_function_dim = value_function_dim
        self.tcn_dim = tcn_dim
        self.decoder_dim = decoder_dim
        self.tcn_buffer_size = tcn_buffer_size

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TCN_mlp(self.features_dim, self.tcn_buffer_size, self.tcn_dim,
                                     self.policy_dim, self.value_function_dim, self.decoder_dim)

    def decode(self, observations: th.Tensor) -> th.Tensor:

        return self.mlp_extractor.decode(observations)
