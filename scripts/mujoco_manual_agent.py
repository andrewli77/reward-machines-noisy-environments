import argparse
import time
import numpy as np
import glfw

import gymnasium as gym
import safety_gym
from utils.env import make_env
from mujoco_py import MjViewer, const


# A simple wrapper for testing SafetyGym-based envs. It uses the PlayViewer that listens to
# key_pressed events and passes the id of the pressed key as part of the observation to the agent.
# (used to control the agent via keyboard)
class PlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.key_pressed = None

    # Shows a text on the upper right corner of the screen
    def show_text(self, text):
        self.env.viewer.show_text(text)

    def render(self, mode='human'):
        if self.env.unwrapped.viewer is None:
            self.env._old_render_mode = 'human'
            self.env.unwrapped.viewer = PlayWrapper.PlayViewer(self.env.unwrapped.sim)
            self.env.unwrapped.viewer.cam.fixedcamid = -1
            self.env.unwrapped.viewer.cam.type = const.CAMERA_FREE
            self.env.unwrapped.viewer.render_swap_callback = self.env.unwrapped.render_swap_callback
            # Turn all the geom groups on
            self.env.unwrapped.viewer.vopt.geomgroup[:] = 1
            self.env.unwrapped._old_render_mode = mode
        self.show_text(self.env.mission_string)
        super().render()

    def wrap_obs(self, obs):
        if not self.env.viewer is None:
            self.key_pressed = self.env.viewer.consume_key()
        return obs

    def reset(self):
        obs, info = self.env.reset()
        return self.wrap_obs(obs), info

    def step(self, action):
        next_obs, original_reward, env_terminated, env_truncated, info = self.env.step(action)
        return self.wrap_obs(next_obs), original_reward, env_terminated, env_truncated, info

    class PlayViewer(MjViewer):
        def __init__(self, sim):
            super().__init__(sim)
            self.key_pressed = None
            self.custom_text = None
            glfw.set_window_size(self.window, 840, 680)
            glfw.set_key_callback(self.window, self.key_callback)

        def show_text(self, text):
            self.custom_text = text

        def consume_key(self):
            ret = self.key_pressed
            self.key_pressed = None
            return ret

        def key_callback(self, window, key, scancode, action, mods):
            self.key_pressed = key
            if action == glfw.RELEASE:
                self.key_pressed = -1

            super().key_callback(window, key, scancode, action, mods)

        def _create_full_overlay(self):
            if (self.custom_text):
                self.add_overlay(const.GRID_TOPRIGHT, "Instruction", self.custom_text)
            step = round(self.sim.data.time / self.sim.model.opt.timestep)
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Step", str(step))
            self.add_overlay(const.GRID_BOTTOMRIGHT, "timestep", "%.5f" % self.sim.model.opt.timestep)

class PlayAgent(object):
    """
    This agent allows user to play with Safety's Point agent.
    Use WASD keys to move.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.prev_act = np.array([0, 0])
        self.last_obs = None
        self.rm_belief = None

    def get_action(self, obs):
        key = self.env.key_pressed

        if(key == glfw.KEY_A):
            current = np.array([0, 0.4])
        elif(key == glfw.KEY_D):
            current = np.array([0, -0.4])
        elif(key == glfw.KEY_W):
            current = np.array([0.1, 0])
        elif(key == glfw.KEY_S):
            current = np.array([-0.1, 0])
        elif(key == -1): # This is glfw.RELEASE
            current = np.array([0, 0])
            self.prev_act = np.array([0, 0])
        else:
            current = np.array([0, 0])

        self.prev_act = np.clip(self.prev_act + current, -1, 1)

        return self.prev_act

def run_policy(agent, env, max_ep_len=None, num_episodes=100, render=True, print_propositions=False):
    o, _ = env.reset()
    n = 0
    ep_ret = 0
    ep_len = 0

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = agent.get_action(o)

        if agent.rm_belief is not None:
            env.viewer.add_overlay(const.GRID_TOPRIGHT, "RM State Belief", str(agent.pretty_print_belief()))

        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, term, trunc, info = env.step(a)

        props = env.get_events()
        if print_propositions and props != "":
            print(props)

        np.set_printoptions(suppress=True)

        ep_ret += r
        ep_len += 1

        if r != 0:
            print("reward: %.3f" %(r))

        if (term or trunc):
            if(r):
                print("SUCCESS", ep_len)
            else:
                print("FAIL", ep_len)

            o, _ = env.reset()
            r, term, trunc, ep_ret, ep_cost, ep_len = 0, False, False, 0, 0, 0
            n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Colour-v0', help='Select the environment to run')
    args = vars(parser.parse_args()) # make it a dictionary

    env = make_env(args["env"], "oracle")
    env.env = PlayWrapper(env.env)

    agent = PlayAgent(env)
    run_policy(agent, env, max_ep_len=50000, num_episodes=1000)

