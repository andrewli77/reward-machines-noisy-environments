import torch
import utils
import random 
import argparse
from tqdm import tqdm
import gymnasium as gym
import envs


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of episodes (default: 1)")

    args = parser.parse_args()

    num_episodes = args.episodes

    # Collect without parallelization
    env = utils.make_env(args.env, "oracle", args.seed)

    obs_space, preprocess_obss = utils.get_obss_preprocessor(env)
    max_episode_length = env.max_steps or 2000

    obss = torch.zeros((num_episodes, max_episode_length, *obs_space['image']))
    events = torch.zeros((num_episodes, max_episode_length, len(env.letter_types)))
    rm_states = torch.zeros((num_episodes, max_episode_length, obs_space['rm_state']))
    actions = torch.zeros((num_episodes, max_episode_length, *env.action_space.shape))
    masks = torch.zeros((num_episodes, max_episode_length))

    for episode in tqdm(range(num_episodes)):
        
        while True:
            obs, _ = env.reset()
            done = False
            step = 0
            returnn = 0
            action = None 
            n_events = 0


            while not done and step < max_episode_length:
                # Add to torch tensors
                preprocessed_obs = preprocess_obss([obs])
                obss[episode, step] = preprocessed_obs.image[0]
                
                labels = env.get_events()

                for event_id in range(len(env.letter_types)):
                    if env.letter_types[event_id] in labels:
                        events[episode, step, event_id] = 1
                        n_events += 1

                rm_states[episode, step] = torch.Tensor(preprocessed_obs.rm_state)
                
                # Semi-random action policy
                if isinstance(env.unwrapped, envs.zone_envs.colour_env.ColourEnv):
                    if step % 10 == 0:
                        action = random.choice([[0.8, 0.], [-0.8, 0.], [0, -0.8], [0, 0.8]])
                elif isinstance(env.unwrapped, envs.minigrid.traffic.TrafficEnv):
                    if step % 10 == 0:
                        action = random.choice([0, 1])
                    elif step % 10 == 9:
                        action = random.choice([0, 1, 2])
                elif isinstance(env.unwrapped, envs.minigrid.kitchen.KitchenEnv):
                    action = env.action_space.sample()

                actions[episode, step] = torch.tensor(action)
                masks[episode, step] = 1 - done # The first 0 is where we no longer have data

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                step += 1

            if n_events >= 1: 
                break
            else:
                obss[episode] = 0.
                events[episode] = 0.
                rm_states[episode] = 0.
                actions[episode] = 0.
                masks[episode] = 0.

    # Save pytorch tensor
    torch.save({"obss": obss,
                "events": events,
                "rm_states": rm_states,
                "actions": actions,
                "masks": masks,
                }, f'collect_{args.env}.pt')