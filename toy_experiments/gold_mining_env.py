from random import *
import numpy as np

# A simple toy gridworld environment to test Reward Machine RL algorithms with noisy detectors.
# ==========================================================================================================
# MAP: (S = start, D = depot, G = gold, F = fool's gold) 
# S . . G
# . F . G 
# . F . G
# D . . G
# ==========================================================================================================
# Objective:  Mine at least one ore of gold and deposit it at the depot. However, the agent cannot
# accurately distinguish real gold from fool's gold, and assigns the following probabilities in its belief
# of whether there is gold at each square.
#
# 0.0  0.0  0.0  0.8
# 0.0  0.3  0.0  0.8
# 0.0  0.6  0.0  0.8
# 0.0  0.0  0.0  0.8
#
# ==========================================================================================================
# States: integers [0,15], representing the agent's location in the grid. The top left is 0, top right is 3,
# bottom left is 12, bottom right is 15. 
#
# Actions: integers [0,4]. (0 = left,1 = right,2 = up,3 = down, 4 = dig).
#
# Reward Machine (task) state:
# 0 = haven't acquired gold
# 1 = acquired gold, but haven't deposited
# 2 = deposited gold


class MiningEnv:
    depot = 12
    has_gold = [False, False, False, True,
                False, False, False, True,
                False, False, False, True,
                False, False, False, True]
    has_gold_model = [0, 0, 0, 0.8,
                        0, 0.3, 0, 0.8,
                        0, 0.6, 0, 0.8,
                        0, 0., 0, 0.8]
    relevant_squares = {3:0, 5:1, 7:2, 11:3, 13:4, 15:5}

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    DIG = 4

    def __init__(self, max_steps=100, movement_cost=0.02):
        self.pos = 0 # Agent's position (0-15)
        self.rm_state = 0 # Task state (0-2)
        self.visited = np.array([False for i in range(6)]) # Keeps track of whether each of the relevant squares was visited
        self.steps = 0 # Counts steps in the current episode
        self.max_steps = max_steps
        self.movement_cost = movement_cost

    def reset(self):
        self.pos = 0
        self.rm_state = 0
        self.visited = np.array([False for i in range(6)])
        self.steps = 0
        return (self.pos, self.rm_state, np.copy(self.visited))

    def step(self, action):
        self.steps += 1 
        reward = 0
        done = False

        if action in [0,1,2,3]:
            reward -= self.movement_cost

        # Left
        if action == self.LEFT:
            if self.pos % 4 != 0:
                self.pos -= 1
        # Right
        elif action == self.RIGHT:
            if self.pos % 4 != 3:
                self.pos += 1
        # Up
        elif action == self.UP:
            if self.pos < 12:
                self.pos += 4
        # Down
        elif action == self.DOWN:
            if self.pos > 3:
                self.pos -= 4
        # Dig
        elif action == self.DIG:
            if self.has_gold[self.pos] and self.rm_state == 0:
                self.rm_state = 1
            if self.pos in self.relevant_squares:
                self.visited[self.relevant_squares[self.pos]] = True

        # Check if we're at the storage
        if self.pos == self.depot:
            done = True
            if self.rm_state == 1:
                self.rm_state = 2
                reward += 1

        if self.steps == self.max_steps:
            done = True

        return (self.pos, self.rm_state, np.copy(self.visited)), reward, done, None
