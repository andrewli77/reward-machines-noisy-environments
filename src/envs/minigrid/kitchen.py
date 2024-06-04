from minigrid.minigrid_env import *
from minigrid.core.world_object import *
from minigrid.core.mission import MissionSpace
import random
from gymnasium.envs.registration import register

class KitchenEnv(MiniGridEnv):
    def __init__(
        self,
        size=9,
        agent_start_pos=None,
        agent_start_dir=0,
        timeout=400,
        is_locked=True,
        randomize_chores=True,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.event_objs = []
        self.target_pos = (1, 1)
        self.is_locked = is_locked
        self.randomize_chores=randomize_chores

        self.letter_types = ['a','b','c','d']

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=timeout,
            see_through_walls=False,
            agent_view_size=7,
            **kwargs,
        )

        self.action_space = gym.spaces.Discrete(3) # Only use left, right, and forward actions

    @staticmethod
    def _gen_mission():
        return "complete the chores and touch the square outside the kitchen"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate inner walls
        self.grid.vert_wall(4, 0)
        self.num_doors = 3
        self.door_poss = [(4, 2), (4, 4), (4, 6)]
        self.before_door_poss = [(i-1, j) for (i, j) in self.door_poss]
        self.before_door_poss.extend([(i+1, j) for (i, j) in self.door_poss])
        self.doors = [Door('yellow', is_locked=self.is_locked) for _ in range(self.num_doors)]
        for i in range(self.num_doors):
            door_pos = self.door_poss[i]
            self.put_obj(self.doors[i], *door_pos)

        self.put_obj(Goal(), *self.target_pos)


        # Place a goal square in the bottom-right corner
        if self.randomize_chores:
            self.chore_poss = []
            chore_left, chore_right, chore_up, chore_down = 5, 7, 1, 7
            while len(self.chore_poss) < 3:
                x = random.randint(chore_left, chore_right)
                y = random.randint(chore_up, chore_down)
                if (x, y) not in self.chore_poss:
                    self.chore_poss.append((x,y))
        else:
            self.chore_poss = [(5, 7), (7, 7), (7, 1)]
        self.chores = [Floor('green'), Floor('blue'), Floor('red')]
        self.event_objs = []
        self.event_objs.append((self.chore_poss[0], 'a'))
        self.event_objs.append((self.chore_poss[1], 'b'))
        self.event_objs.append((self.chore_poss[2], 'c'))
        self.event_objs.append((self.target_pos, 'd'))

        # Randomly complete each goal with 1/3 probability
        for i, pos in enumerate(self.chore_poss):
            done = (random.randint(0,2) == 0)
            if done:
                self.chores[i].color = 'grey'
                if self.event_objs[i][1] not in self.events:
                    self.events += self.event_objs[i][1]


        #self.start_rm_u_id = drop_idx + 1  # a is done, then u_id = 1. etc.

        for pos, obj in zip(self.chore_poss, self.chores):
            self.put_obj(obj, *pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent(top=(1,1), size=(3,7))

    def seed(self, seed):
        random.seed(seed)

    def reset(self, seed=None, options=None):
        self.events = ""
        return super().reset(seed=seed, options=options)

    def step(self, action):
        penalty = 0
        door_opened = False
        # Automatically unlock the door
        if action == self.actions.forward and self.agent_dir == 0:
            for i in range(self.num_doors):
                if tuple(self.agent_pos) == self.before_door_poss[i] and not self.doors[0].is_open:
                    door_opened = True
                    penalty += 0.05
                    for j in range(self.num_doors):
                        self.doors[j].is_open = True
                        self.doors[j].is_locked = False

        next_state, _, terminated, truncated, info = super().step(action)

        for i, pos in enumerate(self.chore_poss):
            if tuple(self.agent_pos) == pos and action == self.actions.forward:
                penalty += 0.05
                self.chores[i].color = 'grey'
                if self.event_objs[i][1] not in self.events:
                    self.events += self.event_objs[i][1]

        if door_opened:
            info['door_opened'] = True
        return next_state, -penalty, terminated, truncated, info

    def get_events(self):
        for pos, event in self.event_objs:
            if tuple(self.agent_pos) == pos and event not in self.events:
                self.events += event
        return self.events

    def get_propositions(self):
        return self.letter_types

register(
    id='Kitchen-v2',
    entry_point='envs.minigrid.kitchen:KitchenEnv'
)