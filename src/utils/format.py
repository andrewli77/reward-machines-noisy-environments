"""
These functions preprocess the observations.
"""

import os
import json
import re
import torch
import torch_ac
import gymnasium as gym
import numpy as np
import utils

from envs import *

def get_obss_preprocessor(env):
    obs_space = env.observation_space
    assert "events" in obs_space.spaces and "features" in obs_space.spaces

    if type(obs_space.spaces["features"])==gym.spaces.Dict and "image" in obs_space.spaces["features"].spaces:
        obs_space = {
            "image": obs_space.spaces["features"]["image"].shape,
            "rm_state": obs_space.spaces["rm-state"].shape[0],
            "events": obs_space.spaces["events"].shape[0]
        }

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["features"]["image"] for obs in obss], device=device),
                "rm_state": preprocess_rm_states([obs["rm-state"] for obs in obss], device=device),
                "events": preprocess_events([obs["events"] for obs in obss], device=device)
            })
    else:
        obs_space = {
            "image": obs_space.spaces["features"].shape,
            "rm_state": obs_space.spaces["rm-state"].shape[0],
            "events": obs_space.spaces["events"].shape[0]
        }

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["features"] for obs in obss], device=device),
                "rm_state": preprocess_rm_states([obs["rm-state"] for obs in obss], device=device),
                "events": preprocess_events([obs["events"] for obs in obss], device=device)
            })
    return obs_space, preprocess_obss

def preprocess_events(events, device=None):
    events = np.array(events)
    return torch.tensor(events, device=device, dtype=torch.float)

def preprocess_rm_states(rm_states, device=None):
    rm_states = np.array(rm_states)
    return torch.tensor(rm_states, device=device, dtype=torch.float)

def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)
