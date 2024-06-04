# Implements simple tabular Q-learning algorithms for the gold mining domain, using linear approximations for generalization.
# Memory is implemented by conditioning on a set of relevant history features: namely, whether the agent has visited each of
# the 6 squares containing either gold or fool's gold.
# You can train the agent in one of two modes:
# - using ground-truth rewards
# - using the agent's predicted rewards 

from random import *
import numpy as np
from gold_mining_env import MiningEnv


# ==========================================================================================================
# ==========================================================================================================
# Base class implementing Q-learning RM algorithms
# ==========================================================================================================
# ==========================================================================================================


class QLRMAlgo:
	def __init__(self,
		reward_type="actual",	# ["actual", "predicted"] 
		discount=0.99, 				# Discount factor γ
		eps=0.2, 					# Random action probability
		lr=0.01,					# Learning rate
		max_frames=1e6,		 
		log_interval=1e4,
	):
		# RL Hyperparameters
		self.reward_type = reward_type
		self.discount = discount
		self.eps = eps
		self.lr = lr
		self.max_frames = max_frames
		self.log_interval = log_interval

		# Initialize envs and logs
		self.initialize_q_table()
		self.env = MiningEnv()
		self.eval_env = MiningEnv()
		self.logs = {'frames': [], 'return': [], 'discounted_return': [], 'predicted_return': [], 'predicted_discounted_return': []}

	# ==========================================================================================================
	# Callable methods
	# ==========================================================================================================
	
	# Train the RM algorithm up to `max_frames` frames. 
	def train(self, print_logs=False):
		eps_num = 0
		frames = 0

		while frames < self.max_frames:
			state = self.env.reset()
			rm_belief = self.initialize_rm_belief()
			eps_num += 1
			eps_len = 0

			# Simulate one episode
			while True:
				# Action selection
				if random() < self.eps:
					action = randint(0,4)
				else:
					action = self.get_best_action(state, rm_belief)

				# Step environment
				next_state, reward, done, _ = self.env.step(action)

				# Update RM belief
				next_rm_belief, predicted_reward = self.update_rm_belief(rm_belief, state, action, next_state)
				if self.reward_type == "predicted":
					if predicted_reward is None:
						raise RuntimeError("Predicted reward is not defined.")
					reward = predicted_reward



				# Update Q-values
				self.update_q_values(state, rm_belief, action, reward, next_state, next_rm_belief, done)

				state = next_state
				rm_belief = next_rm_belief
				eps_len += 1

				# Log results periodically
				if (frames + eps_len) % self.log_interval == 0:
					self.log_policy(frames + eps_len, print_logs)

				if done:
					frames += eps_len
					break


	# Evaluate the current policy over some number of episodes, and log the results
	# in self.logs
	def log_policy(self, frames, print_logs=False):
		returnn, discounted_returnn, predicted_returnn, predicted_discounted_returnn = self.eval_episode()
		self.logs['frames'].append(frames)
		self.logs['return'].append(returnn)
		self.logs['discounted_return'].append(discounted_returnn)
		self.logs['predicted_return'].append(predicted_returnn)
		self.logs['predicted_discounted_return'].append(predicted_discounted_returnn)

		if print_logs:
			print("Frames: %.2f, R: %.2f, Disc R: %.2f -- Pred R: %.2f, Disc Pred R: %.2f"%(frames, returnn, discounted_returnn, predicted_returnn, predicted_discounted_returnn))


	# Evaluate one episode of the current policy.
	def eval_episode(self):
		state = self.eval_env.reset()
		rm_belief = self.initialize_rm_belief() 
		
		eps_len = 0
		returnn = 0
		discounted_returnn = 0
		predicted_returnn = 0
		predicted_discounted_returnn = 0

		while True:
			action = self.get_best_action(state, rm_belief)
			next_state, reward, done, _ = self.eval_env.step(action)
			next_rm_belief, predicted_reward = self.update_rm_belief(rm_belief, state, action, next_state)

			state = next_state
			rm_belief = next_rm_belief

			eps_len += 1
			returnn += reward
			discounted_returnn += reward * self.discount ** (eps_len - 1)

			if predicted_reward is not None:
				predicted_returnn += predicted_reward
				predicted_discounted_returnn += predicted_reward * self.discount ** (eps_len - 1)

			if done or eps_len == 100:
				break
		return returnn, discounted_returnn, predicted_returnn, predicted_discounted_returnn

	# ==========================================================================================================
	# Internal methods
	# ==========================================================================================================

	def get_best_action(self, state, rm_belief):
		best_action = None
		best_q = -99999999999

		for action in range(5):
			qsa = self.get_q_value(state, rm_belief, action)
			if qsa > best_q:
				best_action = action
				best_q = qsa

		return best_action

	def get_state_value(self, state, rm_belief):
		best_value = -99999999999

		for a in range(5):
			best_value = max(best_value, self.get_q_value(state, rm_belief, a))

		return best_value

	# ==========================================================================================================
	# Protocol methods (you need to define these in the subclass)
	# ==========================================================================================================

	# Initializes the Q-value table.
	def initialize_q_table(self):
		raise NotImplementedError()

	# Predict the Q value of the given state-action pair using the values in self.q.
	def get_q_value(self, state, rm_belief, action):
		raise NotImplementedError()

	# Given an experience, update the Q-table.
	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		raise NotImplementedError()

	# Return the initial RM belief
	def initialize_rm_belief(self):
		return None

	# Returns: (next_rm_belief, predicted_reward)
	# Override this in the subclass if needed.
	def update_rm_belief(self, rm_belief, state, action, next_state):
		return None, None

# ==========================================================================================================
# ==========================================================================================================
# Implemented RM algorithms
# ==========================================================================================================
# ==========================================================================================================


# ==========================================================================================================
# QL with the perfect RM state
# ==========================================================================================================
class QLPerfectRM(QLRMAlgo):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)

	def initialize_q_table(self):
		self.q = { (pos, rm_state, a): 0 if rm_state == 2 or pos == MiningEnv.depot else (random() - 0.5) 
						for pos in range(16)
						for rm_state in range(3)
						for a in range(5)}

	def get_q_value(self, state, rm_belief, action):
		return self.q[(state[0], state[1], action)]

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			self.q[(state[0], state[1], action)] += self.lr * (reward - self.q[(state[0], state[1], action)])
		else:
			self.q[(state[0], state[1], action)] += self.lr * (reward + self.discount * self.get_state_value(next_state, None) - self.q[(state[0], state[1], action)])


# ==========================================================================================================
# QL with memory only (no RM)
# ==========================================================================================================
class QLNoRM(QLRMAlgo):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def initialize_q_table(self):
		self.q1 = { (pos, a): 0 if pos == MiningEnv.depot else (random() - 0.5) for pos in range(16) for a in range(5)  }
		self.q2 = { (pos, dug_pos, a): 0 if pos == MiningEnv.depot else (random() - 0.5) for pos in range(16) for dug_pos in range(6) for a in range(5)  }

	def get_q_value(self, state, rm_belief, action):
		q_sum = 0
		q_sum += self.q1[(state[0], action)]
		for dug_pos in range(6):
			q_sum += self.q2[(state[0], dug_pos, action)] * state[2][dug_pos] / 6
		return q_sum

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			delta = reward - self.get_q_value(state, None, action)
		else:
			delta = reward + self.discount * self.get_state_value(next_state, None) - self.get_q_value(state, None, action)

		self.q1[(state[0], action)] += self.lr * delta
		for dug_pos in range(6):
			self.q2[(state[0], dug_pos, action)] += self.lr * delta * state[2][dug_pos] / 6


# ==========================================================================================================
# QL with belief thresholding
# ==========================================================================================================
class QLBeliefThresholding(QLRMAlgo):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def initialize_q_table(self):
		self.q1 = { rm_state : 0 if rm_state == 2 else (random() - 0.5) for rm_state in range(3) }
		self.q2 = { (rm_state, pos, a) : 0 if rm_state == 2 or pos == MiningEnv.depot else (random() - 0.5) for rm_state in range(3) for pos in range(16) for a in range(5) }
		self.q3 = { (rm_state, pos, dug_pos, a) : 0 if rm_state == 2 or pos == MiningEnv.depot else (random() - 0.5) for rm_state in range(3) for pos in range(16) for dug_pos in range(6) for a in range(5) }

	def get_q_value(self, state, rm_belief, action):
		q_sum = 0 

		q_sum += self.q1[rm_belief] + self.q2[(rm_belief, state[0], action)]
		for dug_pos in range(6):
			q_sum += self.q3[(rm_belief, state[0], dug_pos, action)] * state[2][dug_pos] / 6

		return q_sum

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			delta = reward - self.get_q_value(state, rm_belief, action)
		else:
			delta = reward + self.discount * self.get_state_value(next_state, next_rm_belief) - self.get_q_value(state, rm_belief, action)

		self.q1[rm_belief] += self.lr * delta
		self.q2[(rm_belief, state[0], action)] += self.lr * delta

		for dug_pos in range(6):
			self.q3[(rm_belief, state[0], dug_pos, action)] += self.lr * delta * state[2][dug_pos] / 6

	def initialize_rm_belief(self):
		return 0

	def update_rm_belief(self, rm_belief, state, action, next_state):
		predicted_reward = 0

		next_rm_belief = rm_belief
		if rm_belief == 0 and action == MiningEnv.DIG and MiningEnv.has_gold_model[next_state[0]] >= 0.5:
			next_rm_belief = 1
		elif rm_belief == 1 and next_state[0] == MiningEnv.depot:
			next_rm_belief = 2
			predicted_reward = 1

		if action in [0,1,2,3]:
			predicted_reward -= self.env.movement_cost

		return next_rm_belief, predicted_reward


# ==========================================================================================================
# QL with probabilistic updates to the RM belief.
# When the `decorrelate` argument is set to true, the belief is only updated the first time the agent digs
# at one of the relevant squares. 
# ==========================================================================================================
class QLIndependentBelief(QLRMAlgo):
	def __init__(self, decorrelate = False, **kwargs):
		super().__init__(**kwargs)
		self.decorrelate = decorrelate

	def initialize_q_table(self):
		self.q1 = { rm_state : 0 if rm_state == 2 else (random() - 0.5) for rm_state in range(3) }
		self.q2 = { (rm_state, pos, a) : 0 if rm_state == 2 or pos == MiningEnv.depot else (np.random.random() - 0.5) for rm_state in range(3) for pos in range(16) for a in range(5) }
		self.q3 = { (rm_state, pos, dug_pos, a) : 0 if rm_state == 2 or pos == MiningEnv.depot else (np.random.random() - 0.5) for rm_state in range(3) for pos in range(16) for dug_pos in range(6) for a in range(5) }

	def get_q_value(self, state, rm_belief, action):
		q_sum = 0 
		for u in range(3):
			q_sum += self.q1[u] * rm_belief[0][u]
			q_sum += self.q2[(u, state[0], action)] * rm_belief[0][u]
			for dug_pos in range(6):
				q_sum += self.q3[(u, state[0], dug_pos, action)] * rm_belief[0][u] * state[2][dug_pos] / 6
		return q_sum

	def update_q_values(self, state, rm_belief, action, reward, next_state, next_rm_belief, done):
		if done:
			delta = reward - self.get_q_value(state, rm_belief, action)
		else:
			delta = reward + self.discount * self.get_state_value(next_state, next_rm_belief) - self.get_q_value(state, rm_belief, action)

		for u in range(3):
			self.q1[u] += self.lr * delta * rm_belief[0][u]
			self.q2[(u, state[0], action)] += self.lr * delta * rm_belief[0][u]

			for dug_pos in range(6):
				self.q3[(u, state[0], dug_pos, action)] += self.lr * delta * rm_belief[0][u] * state[2][dug_pos] / 6

	def initialize_rm_belief(self):
		dug = [False] * 16
		rm_belief = np.array((1,0,0))
		return rm_belief, dug

	def update_rm_belief(self, rm_belief, state, action, next_state):
		next_pos = next_state[0]
		rm_belief, dug = rm_belief
		predicted_reward = 0

		if next_pos == MiningEnv.depot:
			next_rm_belief = np.array((rm_belief[0], 0, rm_belief[1]))
			predicted_reward = rm_belief[1]
		
		elif action == MiningEnv.DIG:
			if self.decorrelate and dug[next_pos]:
				p1 = rm_belief[1]
			else:
				p1 = rm_belief[1] + rm_belief[0] * MiningEnv.has_gold_model[next_pos]
			next_rm_belief = np.array((1-p1, p1, 0))
			dug[next_pos] = True
		else:
			next_rm_belief = rm_belief

		if action in [0,1,2,3]:
			predicted_reward -= self.env.movement_cost

		return (next_rm_belief, dug), predicted_reward


# ==========================================================================================================
# ==========================================================================================================
# Example of running the code
# ==========================================================================================================
# ==========================================================================================================

if __name__== "__main__":
	algo = QLBeliefThresholding()
	algo.train(print_logs=True)
