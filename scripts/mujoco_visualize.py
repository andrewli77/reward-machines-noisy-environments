import argparse
import time
import numpy as np
import glfw

import gymnasium as gym
import safety_gym
from mujoco_py import MjViewer, const

import utils
from mujoco_manual_agent import run_policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Colour-v0', help='Select the environment to run')
    parser.add_argument("--rm-update-algo",
        default="oracle",
        help="[tdm, naive, ibu, oracle]"
    )
    parser.add_argument("--use-mem", action="store_true", default=False)
    parser.add_argument("--use-mem-detector", action="store_true", default=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int)
    args = vars(parser.parse_args()) # make it a dictionary

    env = utils.make_env(args["env"], args["rm_update_algo"])

    agent = utils.Agent(env, env.observation_space, env.action_space, args["model"],
            args["rm_update_algo"], use_mem=args["use_mem"], use_mem_detector=args["use_mem_detector"])

    run_policy(agent, env, max_ep_len=30000, num_episodes=10, render=True)

