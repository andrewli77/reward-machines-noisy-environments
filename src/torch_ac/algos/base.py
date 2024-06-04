from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from torch_ac.format import default_preprocess_obss
from torch_ac.belief import threshold_rm_beliefs
from torch_ac.utils import DictList, ParallelEnv
import numpy as np


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, detectormodel, device, num_frames_per_proc,
                discount, lr, gae_lambda, entropy_coef, value_loss_coef, recurrence,
                detector_recurrence, preprocess_obss, reshape_reward, rm_update_algo):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        detectormodel : torch.Module
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        recurrence : int
            the number of steps the gradient is propagated back in time
        detector_recurrence : int
            the number of steps the gradient is propagated back in time (for the detector model)
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        use_threshold_rm : bool
        """

        # Store parameters
        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.detectormodel = detectormodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.recurrence = recurrence
        self.detector_recurrence = detector_recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.action_space_shape = envs[0].action_space.shape

        # Control parameters
        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.detectormodel.recurrent or self.detector_recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Configure detectormodel
        self.detectormodel.to(self.device)
        self.detectormodel.train()
        self.rm_update_algo = rm_update_algo
        self.use_rm_belief = (self.rm_update_algo in ["tdm", "naive", "ibu"])

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)
        act_shape = shape + self.action_space_shape

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.rm_beliefs = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        if self.detectormodel.recurrent:
            self.detector_memory = torch.zeros(shape[1], self.detectormodel.memory_size, device=self.device)
            self.detector_memories = torch.zeros(*shape, self.detectormodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*act_shape, device=self.device)#, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*act_shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [1] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            # Perform RM updates...
            with torch.no_grad():
                # Generate a detector belief
                if self.use_rm_belief:
                    if self.detectormodel.recurrent:
                        detector_belief, detector_memory = self.detectormodel(preprocessed_obs, self.detector_memory * self.mask.unsqueeze(1))
                    else:
                        detector_belief = self.detectormodel(preprocessed_obs)

                # tdm: softmax the RM state probabilities
                if self.rm_update_algo == "tdm":
                    detector_belief = F.softmax(detector_belief)

                # naive: use the event predictions to update RM env
                if self.rm_update_algo == "naive":
                    detector_belief = self.env.update_rm_beliefs((detector_belief > 0).cpu().numpy())
                    detector_belief = torch.tensor(np.array(detector_belief), device=self.device, dtype=torch.float)

                # ibu: use the event predictions to update RM env
                if self.rm_update_algo == "ibu":
                    detector_belief = self.env.update_rm_beliefs(detector_belief.cpu().numpy())
                    detector_belief = torch.tensor(np.array(detector_belief), device=self.device, dtype=torch.float)

                # If necessary, add rm_belief to the observation
                if self.use_rm_belief:
                    preprocessed_obs.rm_belief = detector_belief
                    self.rm_beliefs[i] = detector_belief

                # Get policy action
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)

            action = dist.sample()
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            if self.use_rm_belief and self.detectormodel.recurrent:
                self.detector_memories[i] = self.detector_memory
                self.detector_memory = detector_memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        ## Perform RM updates...
        with torch.no_grad():
            ## Generate a detector belief
            if self.use_rm_belief:
                if self.detectormodel.recurrent:
                    detector_belief, detector_memory = self.detectormodel(preprocessed_obs, self.detector_memory * self.mask.unsqueeze(1))
                else:
                    detector_belief = self.detectormodel(preprocessed_obs)

            if self.rm_update_algo == "tdm":
                detector_belief = F.softmax(detector_belief)

            if self.rm_update_algo == "naive":
                detector_belief = self.env.update_rm_beliefs((detector_belief > 0).cpu().numpy())
                detector_belief = torch.tensor(np.array(detector_belief), device=self.device, dtype=torch.float)

            if self.rm_update_algo == "ibu":
                detector_belief = self.env.update_rm_beliefs(detector_belief.cpu().numpy())
                detector_belief = torch.tensor(np.array(detector_belief), device=self.device, dtype=torch.float)

            if self.use_rm_belief:
                preprocessed_obs.rm_belief = detector_belief

            # Get value of next state
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)


        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        if self.detectormodel.recurrent:
            exps.detector_memory = self.detector_memories.transpose(0, 1).reshape(-1, *self.detector_memories.shape[2:])
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape((-1, ) + self.action_space_shape)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        if self.use_rm_belief:
            exps.obs.rm_belief = torch.stack(self.rm_beliefs,dim=0).transpose(0, 1).reshape(self.num_frames, -1)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    def update_parameters(self, exps):
        logs1 = self.update_ac_parameters(exps)
        logs2 = self.update_detector_parameters(exps)
        return {**logs1, **logs2}

    @abstractmethod
    def update_ac_parameters(self, exps):
        pass

    @abstractmethod
    def update_detector_parameters(self, exps):
        pass
