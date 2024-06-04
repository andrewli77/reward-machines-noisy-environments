"""
This is the description of the deep NN currently being used.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from detector_model import BaseDetector
import torch_ac

from gymnasium.spaces import Box, Discrete

from env_model import getEnvModel
from policy_network import PolicyNetwork

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, env, obs_space, action_space, rm_update_algo, hidden_size=64):
        super().__init__()

        # Decide which components are enabled
        self.action_space = action_space

        self.env_model = getEnvModel(env, obs_space, rm_update_algo)
        self.embedding_size = self.env_model.size()
        print("Model: embedding size:", self.embedding_size)
        
        # Define actor's model
        self.actor = PolicyNetwork(self.embedding_size, self.action_space, hiddens=[hidden_size, hidden_size], activation=nn.ReLU())

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = self.env_model(obs)

        # Actor
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

class RecurrentACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, env, obs_space, action_space, rm_update_algo, hidden_size=64):
        super().__init__()

        # Decide which components are enabled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.env_model = getEnvModel(env, obs_space, rm_update_algo)

        # Memory specific code. 
        self.image_embedding_size = self.env_model.size()
        self.memory_rnn = nn.GRUCell(self.image_embedding_size, self.hidden_size)

        print("embedding size:", self.memory_size)

        # Define actor's model
        self.actor = PolicyNetwork(self.memory_size + self.image_embedding_size, self.action_space, hiddens=[hidden_size, hidden_size], activation=nn.ReLU())

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.memory_size + self.image_embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return self.hidden_size

    def forward(self, obs, memory):
        x = self.env_model(obs)
        memory = self.memory_rnn(x, memory)

        # Actor
        embedding = torch.cat([memory, x], dim=1)
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory
