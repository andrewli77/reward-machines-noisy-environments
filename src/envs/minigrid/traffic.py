from minigrid.minigrid_env import *
from minigrid.core.world_object import *
from minigrid.core.mission import MissionSpace
import random
from enum import IntEnum
from gymnasium.envs.registration import register

class Actions1D(IntEnum):
    forward = 0
    backward = 1
    turn180 = 2
    wait = 3

class TrafficEnv(MiniGridEnv):
    """
    Ignorance is not a bliss.
    """
    def __init__(
        self,
        size=13,
        agent_start_pos=(2, 1),
        agent_start_dir=2,
        timeout=100,
        light_remains_green_prob=0.8,
        light_remains_red_prob=0.7,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.target_pos = (1, 1)
        self.event_objs = []
        self.size = size
        self.light_color = 'red'
        self.light_remains_green_prob = light_remains_green_prob
        self.light_remains_red_prob = light_remains_red_prob

        self.letter_types = ['c','d','t']
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=3,
            agent_view_size=7,
            see_through_walls=True,
            max_steps=timeout,
            **kwargs,
        )
        self.actions = Actions1D
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "get the package at the end of the road and return home, without crossing a red light"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        self.step_count = 0

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), *self.target_pos)

        self.light_pos = (self.size//2, 1)
        self.light = Floor('red')  # dummy color
        self.put_obj(self.light, *self.light_pos)
        self.event_objs.append((self.light_pos, 'l'))
        self.update_light_color()

        self.cake_pos = (self.size-2, 1)
        self.cake = Floor('blue')
        self.put_obj(self.cake, *self.cake_pos)
        self.event_objs.append((self.cake_pos, 'c'))

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "cross the traffic light when safe, pick up the cake and return home"

    def seed(self, seed):
        random.seed(seed)

    def step(self, action):
        self.step_count += 1
        self.update_light_color()

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos
        bck_pos = self.back_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        bck_cell = self.grid.get(*bck_pos)

        # Turn 180 deg
        if action == self.actions.turn180:
            self.agent_dir = (self.agent_dir + 2) % 4
        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
        # Move backward
        elif action == self.actions.backward:
            if bck_cell is None or bck_cell.can_overlap():
                self.agent_pos = bck_pos
            if bck_cell is not None and bck_cell.type == "goal":
                terminated = True
        # Wait
        elif action == self.actions.wait:
            pass
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()

        info = {}
        if tuple(self.agent_pos) == self.light_pos and (action != self.actions.forward):
            info['dangerous_light'] = True

        if tuple(self.agent_pos) == self.light_pos and (action == self.actions.forward) and self.shall_not_pass:
            info['red_light'] = True

        if tuple(self.agent_pos) == self.cake_pos:
            info['c'] = True
        if tuple(self.agent_pos) == self.target_pos:
            info['t'] = True

        return obs, reward, terminated, truncated, info

    @property
    def back_pos(self):
        return self.agent_pos - self.dir_vec

    def get_events(self):
        events = ""
        if tuple(self.agent_pos) == self.cake_pos:
            events += 'c'
        if tuple(self.agent_pos) == self.target_pos:
            events += 't'
        if tuple(self.agent_pos) == self.light_pos:
            if self.shall_not_pass:
                events += 'd'
        return events
    
    @property
    def shall_pass(self):
        return self.light_color == 'green' or self.light_color == 'yellow'

    @property
    def shall_not_pass(self):
        return not self.shall_pass

    def update_light_color(self):
        if self.light_color == 'green':
            if random.random() > self.light_remains_green_prob: 
                self.light_color = self.light.color = 'yellow'
        elif self.light_color == 'yellow':
            self.light_color = self.light.color = 'red'
        elif self.light_color == 'red':
            if random.random() > self.light_remains_red_prob: 
                self.light_color = self.light.color = 'green'
        else:
            raise RuntimeError()

    def get_propositions(self):
        return self.letter_types 

register(
    id='Traffic-v0',
    entry_point='envs.minigrid.traffic:TrafficEnv'
)