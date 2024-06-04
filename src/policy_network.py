import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from gymnasium.spaces import Box, Discrete


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, action_space, hiddens=[], scales=None, activation=nn.Tanh()):
        super().__init__()

        layer_dims = [in_dim] + hiddens
        self.action_space = action_space
        self.num_layers = len(layer_dims)
        self.enc_ = nn.Sequential(*[fc(in_dim, out_dim, activation=activation)
            for (in_dim, out_dim) in zip(layer_dims, layer_dims[1:])])

        if (isinstance(self.action_space, Discrete)):
            action_dim = self.action_space.n
            self.discrete_ = nn.Sequential(
                nn.Linear(layer_dims[-1], action_dim)
            )
        elif (isinstance(self.action_space, Box)):
            assert((self.action_space.low == -1).all())
            assert((self.action_space.high == 1).all())
            action_dim = torch.prod(torch.tensor(self.action_space.shape))

            self.mu_ = nn.Linear(layer_dims[-1], action_dim)
            self.std_ = nn.Linear(layer_dims[-1], action_dim)
            self.sigmoid = nn.Sigmoid()
            # self.scales = [1] * action_dim if scales==None else scales
        else:
            print("Unsupported action_space type: ", self.action_space)
            exit(1)

    def forward(self, obs):
        if (isinstance(self.action_space, Discrete)):
            x = self.enc_(obs)
            x = self.discrete_(x)
            return Categorical(logits=F.log_softmax(x, dim=1))
        elif (isinstance(self.action_space, Box)):
            x = self.enc_(obs)
            mu  = 2*(self.sigmoid(self.mu_(x)) - 0.5)
            std = 2*self.sigmoid(self.std_(x)+1.) + 1e-3
            new_shape = (-1,) + self.action_space.shape
            return Normal(mu.reshape(new_shape), std.reshape(new_shape))
        else:
            print("Unsupported action_space type: ", self.action_space)
            exit(1)


def fc(in_dim, out_dim, activation=nn.Tanh()):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation
    )
