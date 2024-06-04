import numpy as np
import copy
import random

import gymnasium as gym
from gymnasium.envs.registration import register
from safety_gym.envs.engine import Engine

# Setup colour dictionary
colours_rgb = {
    'black': (0,0,0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'orange': (255, 165, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'silver': (192, 192, 192),
    'gray': (128, 128, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128),
    'navy': (0, 0, 128),
    'brown': (139, 69, 19)
}

colour_indices = {} 

for i, colour in enumerate(colours_rgb):
    colour_indices[colour] = i

for colour in colours_rgb: 
    colours_rgb[colour] = np.array(colours_rgb[colour]) / 255.

class ColourEnv(Engine):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config['task'] = 'none'

        self._seed = config.pop("seed", None)
        self.seed(self._seed)
        self.rs = np.random.RandomState(seed=self._seed)
        self.letter_types = ['a','b','c','d'] 

        self.pillars_xy = None
        self.hazard_xy = None 
        self.timeout = config['num_steps']
        self.max_steps = self.timeout

        super().__init__(config=config)
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    # Events a,b,c,d mean you're within the range of that pillar. a, b are always the first two pillars
    # in the sequence of pillars you need to visit. Event e means you're within range of the portal. 

    def eventA(self):
        pillar_xy = self.pillars_xy[self.pillar_sequence[0]]
        return self.dist_xy(pillar_xy) < 0.7

    def eventB(self):
        pillar_xy = self.pillars_xy[self.pillar_sequence[1]]
        return self.dist_xy(pillar_xy) < 0.7

    def eventC(self):
        pillar_xy = self.pillars_xy[self.pillar_sequence[2]]
        return self.dist_xy(pillar_xy) < 0.7

    def eventD(self):
        return self.dist_xy(self.hazard_xy) < 0.7

    # Add lidars and rgbs for pillars, lidar for the portal, and colours to the observation space. 
    def build_observation_space(self):
        super().build_observation_space()
        
        for i in range(3):
            self.obs_space_dict.update({f'pillar_{i}_lidar': gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins + 3,), dtype=np.float32)})

        self.obs_space_dict.update({'hazard_0_lidar': gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)})

        self.obs_space_dict.update({'colour': gym.spaces.Box(0.0, 1.0, (len(colours_rgb),), dtype=np.float32)})

        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(self.obs_space_dict)

    def obs(self):
        obs = self.prefill_obs()

        if self.pillars_xy is None:
            self.pillars_xy = [self.data.get_body_xpos(f'pillar{i}').copy() for i in range(3)]

        if self.hazard_xy is None:
            self.hazard_xy = self.data.get_body_xpos('hazard0').copy()

        for i in range(3):
            obs[f'pillar_{i}_lidar'] = np.concatenate([
                self.obs_lidar([self.pillars_xy[i]], 2),
                colours_rgb[self.pillar_colours[i]]])

        obs['hazard_0_lidar'] = self.obs_lidar([self.hazard_xy], 2)

        obs['colour'] = np.zeros(len(colours_rgb))
        obs['colour'][colour_indices[self.colour_1]] = 1.
        obs['colour'][colour_indices[self.colour_2]] = -0.5
        obs['colour'][colour_indices[self.colour_3]] = -1.

        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
        return obs


    def reset(self):
        # Assign each pillar a random colour 
        self.pillar_colours = random.sample(list(colours_rgb.keys()),3)
        self.pillar_sequence = self.rs.permutation(3)
        
        self.colour_1 = self.pillar_colours[self.pillar_sequence[0]]
        self.colour_2 = self.pillar_colours[self.pillar_sequence[1]]
        self.colour_3 = self.pillar_colours[self.pillar_sequence[2]]
        self.mission_string = f"Go to the {self.colour_1} pillar without touching the {self.colour_2} or {self.colour_3} pillars, then enter the portal."
        obs = super().reset()
        
        for i in range(3):
            pillar = self.sim.model.geom_name2id(f'pillar{i}')
            self.sim.model.geom_rgba[pillar][:3] = colours_rgb[self.pillar_colours[i]]

        return obs, {}

    def step(self, a):
        obs, reward, done, info = super().step(a) 
        truncated = self.steps >= self.num_steps
        terminated = done and not truncated
        return obs, reward, terminated, truncated, info


    def get_events(self):
        events = ""

        if self.eventA():
            events += "a"
        if self.eventB():
            events += "b"
        if self.eventC():
            events += "c"
        if self.eventD():
            events += "d"
        return events

    # A hack to get around the observation checker
    def prefill_obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
        if self.observe_goal_lidar:
            obs['goal_lidar'] = self.obs_lidar([self.goal_pos], GROUP_GOAL)
        if self.task == 'push':
            box_pos = self.box_pos
            if self.observe_box_comp:
                obs['box_compass'] = self.obs_compass(box_pos)
            if self.observe_box_lidar:
                obs['box_lidar'] = self.obs_lidar([box_pos], GROUP_BOX)
        if self.task == 'circle' and self.observe_circle:
            obs['circle_lidar'] = self.obs_lidar([self.goal_pos], GROUP_CIRCLE)
        if self.observe_freejoint:
            joint_id = self.model.joint_name2id('robot')
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            assert joint_qposadr == 0  # Needs to be the first entry in qpos
            obs['freejoint'] = self.data.qpos[:7]
        if self.observe_com:
            obs['com'] = self.world.robot_com()
        if self.observe_sensors:
            # Sensors which can be read directly, without processing
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.hinge_vel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballangvel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            # Process angular position sensors
            if self.sensors_angle_components:
                for sensor in self.robot.hinge_pos_names:
                    theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                    obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
                for sensor in self.robot.ballquat_names:
                    quat = self.world.get_sensor(sensor)
                    obs[sensor] = quat2mat(quat)
            else:  # Otherwise read sensors directly
                for sensor in self.robot.hinge_pos_names:
                    obs[sensor] = self.world.get_sensor(sensor)
                for sensor in self.robot.ballquat_names:
                    obs[sensor] = self.world.get_sensor(sensor)
        if self.observe_remaining:
            obs['remaining'] = np.array([self.steps / self.num_steps])
            assert 0.0 <= obs['remaining'][0] <= 1.0, 'bad remaining {}'.format(obs['remaining'])
        if self.walls_num and self.observe_walls:
            obs['walls_lidar'] = self.obs_lidar(self.walls_pos, GROUP_WALL)
        if self.observe_hazards:
            obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, GROUP_HAZARD)
        if self.observe_vases:
            obs['vases_lidar'] = self.obs_lidar(self.vases_pos, GROUP_VASE)
        if self.gremlins_num and self.observe_gremlins:
            obs['gremlins_lidar'] = self.obs_lidar(self.gremlins_obj_pos, GROUP_GREMLIN)
        if self.pillars_num and self.observe_pillars:
            obs['pillars_lidar'] = self.obs_lidar(self.pillars_xy, GROUP_PILLAR)
        if self.buttons_num and self.observe_buttons:
            # Buttons observation is zero while buttons are resetting
            if self.buttons_timer == 0:
                obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, GROUP_BUTTON)
            else:
                obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)
        if self.observe_qpos:
            obs['qpos'] = self.data.qpos.copy()
        if self.observe_qvel:
            obs['qvel'] = self.data.qvel.copy()
        if self.observe_ctrl:
            obs['ctrl'] = self.data.ctrl.copy()
        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        return obs

colour_config = {
    'robot_base': 'xmls/point.xml',
    'observe_remaining': True,
    'num_steps': 2000,
    'robot_locations': [(-0.2,0)],
    'placements_extents': [-2., -2., 2., 2.],
    'pillars_num': 3,
    'pillars_locations': [(-1.6, -1.6), (1.2, 1.3), (-1.4, 0.)],
    'hazards_num': 1,
    'hazards_locations': [(1.2, -0.6)],
    'observation_flatten': True,
}

register(
    id='Colour-v0',
    entry_point='envs.zone_envs.colour_env:ColourEnv',
    kwargs={'config': colour_config})