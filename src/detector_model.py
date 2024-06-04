from abc import ABC, abstractmethod
from torch import nn
import torch.nn.functional as F
import torch
from envs import *
from minigrid.minigrid_env import MiniGridEnv

def getDetectorModel(env, obs_space, rm_update_algo, use_mem_detector):
    env = env.unwrapped
    detectormodel = None

    if use_mem_detector:
        if rm_update_algo in ["naive", "ibu"]:
            if isinstance(env, MiniGridEnv):
                detectormodel = RecurrentMinigridDetectorModel(obs_space, obs_space['events'])
            if isinstance(env, ColourEnv):
                detectormodel = RecurrentColourDetectorModel(obs_space, obs_space['events'])

        elif rm_update_algo in ["tdm"]:
            if isinstance(env, MiniGridEnv):
                detectormodel = RecurrentMinigridDetectorModel(obs_space, obs_space['rm_state'])
            if isinstance(env, ColourEnv):
                detectormodel = RecurrentColourDetectorModel(obs_space, obs_space['rm_state'])

        elif rm_update_algo in ["hybrid"]:
            if isinstance(env, MiniGridEnv):
                raise NotImplementedError()
            if isinstance(env, ColourEnv):
                detectormodel = HybridColourDetectorModel(obs_space, obs_space['rm_state'], obs_space['events'])

        elif rm_update_algo in ["oracle", "no_rm"]:
            detectormodel = PerfectDetector(obs_space)
        else:
            raise NotImplementedError()
    else:
        if rm_update_algo in ["oracle", "no_rm"]:
            detectormodel = PerfectDetector(obs_space)

        elif rm_update_algo in ["naive", "ibu"]:
            if isinstance(env, ColourEnv):
                detectormodel = ColourDetectorModel(obs_space, obs_space['events'])
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    return detectormodel

class BaseDetector(ABC, nn.Module):
    recurrent = False
    params_free = False
    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def forward(self, obs):
        """
        Returns probabilities over rm states (normalized)
        """
        pass

class PerfectDetector(BaseDetector):
    params_free = True
    def __init__(self, obs_space):
        super().__init__()
        self.out_dim = obs_space['rm_state']

    def forward(self, obs):
        return obs.rm_state

###### MiniGrid ########
########################
########################
########################

class RecurrentMinigridDetectorModel(BaseDetector):
    recurrent = True

    def __init__(self, obs_space, out_dim) -> None:
        super().__init__()
        n, m, k = obs_space['image']
        self.out_dim = out_dim
        self.image_embedding_size = 64
        
        # image -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(n*m*k, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.image_embedding_size)
        )

        # embedding -> embedding
        self.memory_rnn = nn.GRUCell(self.image_embedding_size, self.memory_size)

        # embedding -> rm states
        self.decoder = nn.Linear(self.memory_size, out_dim)

    @property
    def memory_size(self):
        return 64

    def forward(self, obs, memory):
        x = self.encoder(obs.image.flatten(start_dim=-3))
        memory = self.memory_rnn(x, memory)
        out = self.decoder(memory)

        return out, memory


######## MuJoCo ########
########################
########################
########################

class RecurrentColourDetectorModel(BaseDetector):
    recurrent = True

    def __init__(self, obs_space, out_dim) -> None:
        super().__init__()

        input_dim = obs_space['image'][0]
        self.out_dim = out_dim
        self.image_embedding_size = 128
        
        # image -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_embedding_size),
        )

        # image_embedding -> embedding
        self.memory_rnn1 = nn.GRUCell(self.image_embedding_size, self.semi_memory_size)
        self.memory_rnn2 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)

        # embedding -> predictions
        self.decoder = nn.Sequential(
            nn.Linear(self.memory_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.out_dim)
        )

    @property
    def memory_size(self):
        return self.semi_memory_size * 2

    @property
    def semi_memory_size(self):
        return 128

    def forward(self, obs, memory):
        memory1, memory2 = (memory[:, :self.semi_memory_size],
            memory[:, self.semi_memory_size:2*self.semi_memory_size])
        
        x = self.encoder(obs.image)
        memory1 = self.memory_rnn1(x, memory1)
        memory2 = self.memory_rnn2(F.relu(memory1), memory2)
        memory = torch.cat([memory1, memory2], dim=1)

        out = self.decoder(memory)

        return out, memory

# First predicts the events at the current time, and uses this to predict the RM state distribution
# This is an alternative to predicting only RM states. 
class HybridColourDetectorModel(BaseDetector):
    recurrent = True

    def __init__(self, obs_space, n_rm_states, n_events) -> None:
        super().__init__()

        input_dim = obs_space['image'][0]
        self.n_rm_states = n_rm_states
        self.n_events = n_events
        self.image_embedding_size = 128
        
        # image -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_embedding_size),
        )

        self.decoder_events = nn.Sequential(
            nn.Linear(self.image_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_events)
        )

        # image_embedding -> embedding
        self.memory_rnn1 = nn.GRUCell(self.image_embedding_size + self.n_events, self.semi_memory_size)
        self.memory_rnn2 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)

        # embedding -> rm states
        self.decoder_rm_states = nn.Sequential(
            nn.Linear(self.memory_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_rm_states)
        )

    @property
    def memory_size(self):
        return self.semi_memory_size * 2

    @property
    def semi_memory_size(self):
        return 128

    def forward(self, obs, memory):
        memory1, memory2 = (memory[:, :self.semi_memory_size],
            memory[:, self.semi_memory_size:2*self.semi_memory_size])
        
        x = self.encoder(obs.image)
        out_events = self.decoder_events(x)

        memory1 = self.memory_rnn1(torch.cat([x, out_events], dim=1), memory1)
        memory2 = self.memory_rnn2(F.relu(memory1), memory2)
        memory = torch.cat([memory1, memory2], dim=1)

        out_rm_states = self.decoder_rm_states(memory)

        return (out_events, out_rm_states), memory

class ColourDetectorModel(BaseDetector):
    recurrent = False

    def __init__(self, obs_space, out_dim) -> None:
        super().__init__()

        input_dim = obs_space['image'][0]
        self.out_dim = out_dim
        self.image_embedding_size = 128
        
        # image -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, obs):
        return self.encoder(obs.image)
