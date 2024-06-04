import argparse
import numpy 
import torch
import time
import utils
from utils.env import make_env
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--use-mem", action="store_true", default=False)
parser.add_argument("--use-mem-detector", action="store_true", default=False)
parser.add_argument("--rm-update-algo", type=str, default="tdm",
                    help="[tdm, naive, ibu, oracle, no_rm]")

parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")
parser.add_argument("--no-render", action="store_true", default=False)
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")


args = parser.parse_args()

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment
env = make_env(
    args.env,
    args.rm_update_algo,
    args.seed,
    render_mode="human",
    screen_size=640
)
env.reset()
print("Environment loaded\n")

# Load agent
model_dir = args.model
agent = utils.Agent(env, env.observation_space, env.action_space, model_dir, args.rm_update_algo, hidden_size=64, use_mem=args.use_mem, use_mem_detector=args.use_mem_detector, device=device)
print("Agent loaded\n")

# Create a window to view the environment
if not args.no_render:
    env.env.render()

if args.gif:
    from array2gif import write_gif
    frames = []


episode_returns = []

for episode in range(args.episodes):
    returnn = 0
    obs, _ = env.reset()

    i = 0
    while True:
        if not args.no_render:
            env.env.render()
            time.sleep(args.pause)

        if args.gif:
            frames.append(numpy.moveaxis(env.env.get_frame(), 2, 0))

        action = agent.get_action(obs)

        if agent.rm_belief is not None:
            env.unwrapped.mission = str(agent.pretty_print_belief()) #viewer.add_overlay(const.GRID_TOPRIGHT, "RM State Belief", str(agent.pretty_print_belief()))

        obs, reward, terminated, truncated, info = env.step(action)
        agent.analyze_feedback(terminated | truncated)
        returnn += reward 

        i += 1
        if terminated | truncated or i == 50:
            print("Episode %i --- Return: %.3f --- Num Steps: %d" %(episode+1, returnn, i))
            episode_returns.append(returnn)
            break

print("Average return:", sum(episode_returns) / len(episode_returns))

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")