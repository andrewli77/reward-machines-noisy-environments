import torch
import torch.nn as nn

from envs import *

def getEnvModel(env, obs_space, rm_update_algo):
    env = env.unwrapped

    if isinstance(env, MiniGridEnv):
        return MinigridEnvModel(obs_space, rm_update_algo)
    if isinstance(env, ColourEnv):
        if 'vision' in obs_space:
            return ColourVisionEnvModel(obs_space)
        else:
            return ColourEnvModel(obs_space)

    # The default case (No environment observations) - SimpleLTLEnv uses this
    return EnvModel(obs_space)


"""
This class is in charge of embedding the environment part of the observations.
Every environment has its own set of observations ('image', 'direction', etc) which is handeled
here by associated EnvModel subclass.

How to subclass this:
    1. Call the super().__init__() from your init
    2. In your __init__ after building the compute graph set the self.embedding_size appropriately
    3. In your forward() method call the super().forward as the default case.
    4. Add the if statement in the getEnvModel() method
"""
class EnvModel(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.embedding_size = 0

    def forward(self, obs):
        return None

    def size(self):
        return self.embedding_size

class MinigridEnvModel(EnvModel):
    def __init__(self, obs_space, rm_update_algo):
        super().__init__(obs_space)
        self.image_embedding_size = 64
        self.embedding_size = self.image_embedding_size
        self.rm_update_algo = rm_update_algo
        if self.rm_update_algo != "no_rm":
            self.embedding_size += obs_space["rm_state"]

        # Image based
        n, m, k = obs_space["image"]
        self.image_conv = nn.Sequential(
            nn.Linear(n*m*k, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.image_embedding_size)
        )

    def forward(self, obs):
        x = self.image_conv(obs.image.flatten(start_dim=1))
        if self.rm_update_algo == "no_rm":
            embedding = x
        elif self.rm_update_algo in ["tdm", "ibu", "naive"]:
            y = obs.rm_belief
            embedding = torch.cat((x, y), dim=1)
        elif self.rm_update_algo == "oracle":
            y = obs.rm_state
            embedding = torch.cat((x, y), dim=1)
        else:
            raise NotImplementedError()
        return embedding

class ColourEnvModel(EnvModel):
    def __init__(self, obs_space, no_rm=False):
        super().__init__(obs_space)
        self.no_rm = no_rm

        input_dim = obs_space["image"][0]
        
        if not no_rm:
            input_dim += obs_space["rm_state"]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.embedding_size = 128

    def forward(self, obs, use_rm_belief=False):
        if self.no_rm:
            _input = obs.image
        elif use_rm_belief:
            _input = torch.cat((obs.image, obs.rm_belief), dim=1)
        else:
            _input = torch.cat((obs.image, obs.rm_state), dim=1)
        return self.mlp(_input)

class ColourVisionEnvModel(EnvModel):
    def __init__(self, obs_space, no_rm=False):
        super().__init__(obs_space)
        self.no_rm = no_rm
        self.embedding_size = 256
        
        self.image_conv = nn.Sequential( # 3 x 40 x 60
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(), # 16 x 40 x 60
            nn.MaxPool2d((2,3)), # 16 x 20 x 20
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(), # 32 x 18 x 18
            nn.MaxPool2d(2), # 32 x 9 x 9
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(), # 32 x 7 x 7
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(), # 32 x 5 x 5
            nn.Flatten()
        )
        image_embedding_size = 800
        input_size = image_embedding_size + obs_space['obs']
        if not no_rm:
            input_size += obs_space["rm_state"]
        
        self.mlp = nn.Sequential(
                nn.Linear(input_size, self.embedding_size),
                nn.ReLU()
        )

    def forward(self, obs, use_rm_belief=False):
        x = self.image_conv(obs.vision.transpose(1, 3).transpose(2, 3))
        x = torch.cat([x, obs.obs], dim=1)

        if self.no_rm:
            embedding = x
        elif use_rm_belief:
            y = obs.rm_belief
            embedding = torch.cat((x, y), dim=1)
        else:
            y = obs.rm_state
            embedding = torch.cat((x, y), dim=1)
        return self.mlp(embedding) 
