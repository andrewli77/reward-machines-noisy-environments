"""
These are simple wrappers that will include RMs to any given environment.
It also keeps track of the RM state as the agent interacts with the envirionment.

However, each environment must implement the following function:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.

Notes:
    - The episode ends if the RM reaches a terminal state or the environment reaches a terminal state.
    - The agent only gets the reward given by the RM.
    - Rewards coming from the environment are ignored.

Extra notes:
This file was originally from 
https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/rm_environment.py
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from reward_machines import RewardMachine
from torch_ac.belief import possible_true_propositions

class RewardMachineEnv(gymnasium.Wrapper):
    def __init__(self, env, rm_files):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve a different RM task
            - This code keeps track of the current state on the current RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of strings with paths to the RM files.
        """
        super().__init__(env)

        # Loading the reward machines
        self.rm_files = rm_files
        self.reward_machines = []
        self.num_rm_states = 0
        for rm_file in rm_files:
            rm = RewardMachine(rm_file)
            self.num_rm_states += len(rm.get_states())
            self.reward_machines.append(rm)
        self.num_rms = len(self.reward_machines)
        self.n_propositions = len(self.env.letter_types)
        self.proposition_to_idx = {prop:i for i, prop in enumerate(self.env.letter_types)}

        # The observation space is a dictionary including the env features and a one-hot representation of the state in the reward machine
        self.observation_space = spaces.Dict(
            {
                'features': env.observation_space,
                'rm-state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8),
                'events': spaces.Box(low=0, high=1, shape=(self.n_propositions,), dtype=np.uint8),
            })

        # Computing one-hot encodings for the non-terminal RM states
        self.rm_state_features = {}
        for rm_id, rm in enumerate(self.reward_machines):
            for u_id in rm.get_states():
                u_features = np.zeros(self.num_rm_states)
                u_features[len(self.rm_state_features)] = 1
                self.rm_state_features[(rm_id,u_id)] = u_features
        self.rm_done_feat = np.zeros(self.num_rm_states) # for terminal RM states, we give as features an array of zeros

        # Selecting the current RM task
        self.current_rm_id = -1
        self.current_rm    = None

    def reset(self):
        # Reseting the environment and selecting the next RM tasks
        self.obs, info = self.env.reset()
        self.current_rm_id = (self.current_rm_id+1)%self.num_rms
        self.current_rm    = self.reward_machines[self.current_rm_id]
        self.current_u_id  = self.current_rm.reset()

        true_props = self.env.get_events()

        # Adding the RM state to the observation
        return self.get_observation(self.obs, self.current_rm_id, self.current_u_id, False, true_props), info

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_terminated, env_truncated, info = self.env.step(action)
        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()
        self.crm_params = self.obs, action, next_obs, env_terminated, env_truncated, true_props, info
        self.obs = next_obs

        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info)
        # returning the result of this action
        terminated = rm_done or env_terminated
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_u_id, terminated, true_props)

        return rm_obs, rm_rew + original_reward, terminated, env_truncated, info

    def get_observation(self, next_obs, rm_id, u_id, done, true_props):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id,u_id)]
        events = np.zeros(self.n_propositions)

        for prop in true_props:
            events[self.proposition_to_idx[prop]] = 1

        rm_obs = {'features': next_obs,'rm-state': rm_feat, 'events': events}
        return rm_obs

    def render(self):
        return self.env.render()      


# Modifies a reward machine env in the following ways:
# -> Events are returned as part of the observation:   obs['events']
# -> After resetting, or stepping and receiving `next_obs`, the function update_rm_belief(event_preds(next_obs)) MUST be called (this is checked)
# -----> Order of calls should be: o1, update_rm_belief, step, update_rm_belief, step, ...
# -> `event_preds` is a 0-1 vector predicting the truth values of events. Internally, the predicted events are used to progress a belief reward machine state `u_pred`. 
# -> `u_pred` is returned as part of the observation, NOT the true reward machine state. HOWEVER, the true reward machine state is still used to provide rewards.
class RewardMachineNoisyThresholdEnv(RewardMachineEnv):
    def __init__(self, env, rm_files):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve a different RM task
            - This code keeps track of the current state on the current RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of strings with paths to the RM files.
        """
        super().__init__(env, rm_files)
        self.n_propositions = len(self.env.letter_types)
        self.proposition_to_idx = {prop:i for i, prop in enumerate(self.env.letter_types)}

        self.observation_space = spaces.Dict(
            {
                'features': env.observation_space,
                'rm-state': spaces.Box(low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8),
                'events': spaces.Box(low=0, high=1, shape=(self.n_propositions,), dtype=np.uint8)
            })

        self.last_reset_or_stepped = False


    def reset(self):
        self.last_reset_or_stepped = True

        # Reseting the environment and selecting the next RM tasks
        self.obs, info = self.env.reset()
        self.current_rm_id = (self.current_rm_id+1)%self.num_rms
        self.current_rm    = self.reward_machines[self.current_rm_id]
        self.current_u_id  = self.current_rm.reset()

        self.belief_u_id = self.current_u_id
        true_props = self.env.get_events()
        # Adding the RM state to the observation
        return self.get_observation(self.obs, self.current_rm_id, self.current_u_id, False, true_props), info

    def step(self, action):
        assert not self.last_reset_or_stepped, "Step called before updating RM belief state -- call `update_rm_beliefs`!"
        self.last_reset_or_stepped = True

        # executing the action in the environment
        next_obs, original_reward, env_terminated, env_truncated, info = self.env.step(action)

        # getting the output of the detectors and saving information for generating counterfactual experiences
        true_props = self.env.get_events()

        self.crm_params = self.obs, action, next_obs, env_terminated, true_props, info
        self.obs = next_obs

        # update the RM state
        self.current_u_id, rm_rew, rm_done = self.current_rm.step(self.current_u_id, true_props, info)

        # returning the result of this action
        terminated = rm_done or env_terminated
        rm_obs = self.get_observation(next_obs, self.current_rm_id, self.current_u_id, terminated, true_props)

        return rm_obs, rm_rew + original_reward, terminated, env_truncated, info

    def update_rm_beliefs(self, event_preds):
        assert(self.last_reset_or_stepped, "Need to step or reset before updating the RM belief state again!")
        self.last_reset_or_stepped = False

        assert(len(event_preds) == self.n_propositions)
        true_props = ""

        if self.belief_u_id == -1:
            return self.rm_done_feat

        for i, prop in enumerate(self.env.letter_types):
            if event_preds[i]:
                true_props += prop

        self.belief_u_id, _ = self.current_rm.get_next_state(self.belief_u_id, true_props)
        if self.belief_u_id == -1:
            return self.rm_done_feat
        return self.rm_state_features[(self.current_rm_id, self.belief_u_id)]

    def get_observation(self, next_obs, rm_id, u_id, done, true_props):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[(rm_id,u_id)]
        events = np.zeros(self.n_propositions)

        for prop in true_props:
            events[self.proposition_to_idx[prop]] = 1

        rm_obs = {'features': next_obs,'rm-state': rm_feat, 'events': events}
        return rm_obs              

# This wrapper is a twin of RewardMachineNoisyThresholdEnv. Noticeable differences are:
# - `event_preds` is a vector of (binary) logits from the detector model.
# - `u_pred` is a normalized vector instead of an integer.
# - the engine of independent belief update is implemented here.
# - self.belief_u_id is replaced with self.belief_u_dist
class RewardMachineNoisyBeliefUpdateEnv(RewardMachineNoisyThresholdEnv):
    def __init__(self, env, rm_files):
        """
        RM environment
        --------------------
        It adds a set of RMs to the environment:
            - Every episode, the agent has to solve a different RM task
            - This code keeps track of the current state on the current RM task
            - The id of the RM state is appended to the observations
            - The reward given to the agent comes from the RM

        Parameters
        --------------------
            - env: original environment. It must implement the following function:
                - get_events(...): Returns the propositions that currently hold on the environment.
            - rm_files: list of strings with paths to the RM files.
        """
        super().__init__(env, rm_files)

    def reset(self):
        obs, info = super().reset()
        del self.belief_u_id
        self.belief_u_dist = [1.0 if u_id == self.current_u_id else 0.0 for u_id in range(self.num_rm_states)]
        return obs, info

    # self.step(action) inherents RewardMachineNoisyThresholdEnv
    # self.get_observation() inherents RewardMachineNoisyThresholdEnv

    def update_rm_beliefs(self, event_preds, logit=True):
        """
        Input: event_preds is a vector tensor of (binary) logits from the detector model
        Output: a weighed overlay/sum of rm_state_features
        """

        assert(self.last_reset_or_stepped, "Need to step or reset before updating the RM belief state again!")
        self.last_reset_or_stepped = False

        assert(len(event_preds) == self.n_propositions)

        if logit is True:
            noisy_props = { prop : 1./(1.+ np.exp(-event_preds[idx]))
                for prop, idx in self.proposition_to_idx.items() }
        else:
            noisy_props = { prop : event_preds[idx]
                for prop, idx in self.proposition_to_idx.items()}

        belief_u_dist = [0.] * self.num_rm_states
        for u_id, u_prob in enumerate(self.belief_u_dist):
            for true_props, true_props_prob in possible_true_propositions(noisy_props):
                prob = u_prob * true_props_prob
                next_u_id = self.current_rm.get_next_state(u_id, true_props)[0]
                belief_u_dist[next_u_id] += prob
        self.belief_u_dist = belief_u_dist

        aggr_feat = sum(prob * self.rm_state_features[(self.current_rm_id, u_id)]
            for u_id, prob in enumerate(belief_u_dist))
        return aggr_feat

