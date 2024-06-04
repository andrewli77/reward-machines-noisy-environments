# # Manually control an agent using keyboard inputs. You can use this to test an environment.
# # Taken from https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/manual_control.py 

#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
import envs.minigrid.kitchen, envs.minigrid.traffic
from utils.env import make_env
import envs

class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}, events={self.env.get_events()}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset()
        self.env.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )

    args = parser.parse_args()

    env = make_env(
        args.env_id,
        'oracle',
        tile_size=32,
        render_mode="human",
        agent_pov=args.agent_view,
        screen_size=640,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env.env = RGBImgPartialObsWrapper(env.env, 32)
        env.env = ImgObsWrapper(env.env)

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()