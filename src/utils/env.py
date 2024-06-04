"""
This class defines the environments that we are going to use.
Note that this is the place to include the right LTL-Wrapper for each environment.
"""


from os import listdir
import gymnasium as gym
import rm_wrappers
from os.path import join

key2rmfiles = {
    'Kitchen-v2':['k1.txt'],
    'Traffic-v0':['f1.txt'],
    'Colour-v0':['p1.txt'],
}

def make_env(env_key, rm_update_algo, seed=None, **kwargs):
    rm_files_dir = 'src/envs/rm_files'
    rm_files = [join(rm_files_dir, fname) for fname in key2rmfiles[env_key]]
    env = gym.make(env_key, **kwargs)

    if rm_update_algo == "naive":
        env = rm_wrappers.RewardMachineNoisyThresholdEnv(env, rm_files)
    elif rm_update_algo == "ibu":
        env = rm_wrappers.RewardMachineNoisyBeliefUpdateEnv(env, rm_files)
    else:
        env = rm_wrappers.RewardMachineEnv(env, rm_files)
    env.seed(seed)
    
    return env