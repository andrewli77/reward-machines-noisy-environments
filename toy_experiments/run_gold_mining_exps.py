from gold_mining_algos import *
import pickle as pkl

def run_and_save(algo, file_name):
	algo.train(print_logs=True)
	with open(file_name, "wb") as file:
		pkl.dump(algo.logs, file)

# Training on ground-truth rewards but imperfect labels
algo = QLIndependentBelief(decorrelate=True)
run_and_save(algo, "toy_experiments/data/policy_learning/tdm_%d.pkl"%(i+1))

# Training on predicted rewards
algo = QLIndependentBelief(decorrelate=True, reward_type="predicted")
run_and_save(algo, "toy_experiments/data/internal_rewards/tdm_%d.pkl"%(i+1))
