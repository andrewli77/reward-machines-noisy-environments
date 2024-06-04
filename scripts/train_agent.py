import argparse
import time
import datetime
import torch
import torch_ac
import sys
import wandb
import os
import glob

import utils
from model import ACModel, RecurrentACModel
from detector_model import *

# Parse arguments
parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=20,
                    help="number of updates between two saves (default: 20, 0 means no saving)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 1)")
parser.add_argument("--frames", type=int, default=5*10**7,
                    help="number of frames of training (default: 5*10e8)")
parser.add_argument("--load-model", default=None,
                    help="Directory of the model to load")
parser.add_argument("--wandb", action="store_true", default=False,
                    help="Log the experiment with weights & biases")

## Parameters for PPO algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=100,
                    help="batch size for PPO (default: 100)")
parser.add_argument("--frames-per-proc", type=int, default=100,
                    help="number of frames per process before update")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.001,
                    help="entropy term coefficient (default: 0.001)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=4,
                    help="number of time-steps gradient is backpropagated (default: 4). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--hidden-size", type=int, default=128,
                    help="hidden dimensions for actor-critic.")

## Parameters for training the detector
parser.add_argument("--detector-epochs", type=int, default=4,
                    help="number of epochs for detector (default: 4)")
parser.add_argument("--detector-batch-size", type=int, default=100,
                    help="batch size for detector (default: 100)")
parser.add_argument("--detector-lr", type=float, default=0.0003,
                    help="learning rate for detector (default: 0.0003)")
parser.add_argument("--detector-recurrence", type=int, default=4,
        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the detector model to have memory.")
parser.add_argument("--rm-update-algo", type=str, default="tdm",
                    help="[tdm, naive, ibu, oracle, no_rm]")
args = parser.parse_args()

use_mem = args.recurrence > 1
use_mem_detector = args.detector_recurrence > 1

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
name = args.rm_update_algo
recurrent = "-r" if use_mem else ""
default_model_name = f"{name}{recurrent}-{args.env}-seed{args.seed}"
pretrained_model_name = f"{name}-{args.env}-pretrained"
model_name = default_model_name
storage_dir = "storage"
model_dir = utils.get_model_dir(model_name, storage_dir)
in_model_dir = None if args.load_model is None else utils.get_model_dir(args.load_model, "")

# Load loggers
txt_logger = utils.get_txt_logger(model_dir + "/train")
csv_file, csv_logger = utils.get_csv_logger(model_dir + "/train")

if not args.wandb:
    os.environ['WANDB_MODE'] = 'disabled'
wandb.init(project='noisy-detector')
wandb.run.name = default_model_name
wandb.run.save()
config = wandb.config
config.update(args)
utils.save_config(model_dir + "/train", args)

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments
envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.env, args.rm_update_algo, args.seed + 10000*i))
txt_logger.info("Environments loaded\n")

# Load training status
if in_model_dir:
    status = utils.get_status(in_model_dir)
else:
    try:
        status = utils.get_status(model_dir + "/train")
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded.\n")


# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0])
if "vocab" in status and preprocess_obss.vocab is not None:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded.\n")


# Load model
detectormodel = getDetectorModel(envs[0], obs_space, args.rm_update_algo, use_mem_detector)

if use_mem:
    acmodel = RecurrentACModel(envs[0], obs_space, envs[0].action_space, args.rm_update_algo, args.hidden_size)
else:
    acmodel = ACModel(envs[0], obs_space, envs[0].action_space, args.rm_update_algo, args.hidden_size)

# Load actor-critic model from checkpoint if it exists
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
    txt_logger.info("Loading acmodel from existing run.\n")

# Load detector model from checkpoint if it exists
if "detector_model_state" in status:
    detectormodel.load_state_dict(status["detector_model_state"])
    txt_logger.info("Loading detector model from existing run.\n")
# Otherwise, load detector model from a pretrained source if it exists
elif args.rm_update_algo not in ["oracle", "no_rm"]:
    try:
        pretrain_detector_status = utils.get_status(utils.get_model_dir(pretrained_model_name, storage_dir))
        if "model_state" in pretrain_detector_status:
            detectormodel.load_state_dict(pretrain_detector_status["model_state"])
            txt_logger.info("Loading detector model from pretraining run.\n")
    except OSError:
        pass

acmodel.to(device)
txt_logger.info("AC Model loaded.\n")
txt_logger.info("{}\n".format(acmodel))

detectormodel.to(device)
txt_logger.info("Detector Model loaded.\n")
txt_logger.info("{}\n".format(detectormodel))

# Load algo
algo = torch_ac.PPOAlgo(envs, acmodel, detectormodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                        args.entropy_coef, args.value_loss_coef, args.recurrence, 
                        args.detector_epochs, args.detector_batch_size, args.detector_lr, args.detector_recurrence,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, rm_update_algo=args.rm_update_algo)

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Loading optimizer from existing run.\n")

if "optimizer_detector_state" in status:
    algo.optimizer_detector.load_state_dict(status["optimizer_detector_state"])
    txt_logger.info("Loading detector optimizer from existing run.\n")

txt_logger.info("Optimizer loaded.\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters
    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()


    if args.rm_update_algo in ["oracle", "no_rm"] or args.detector_epochs == 0:
        logs3 = {
            'detector_loss_train': 0,
            'detector_grad_train': 0,
            'detector_top1_accuracy_train': 0,
            'detector_loss_test': 0,
            'detector_grad_test': 0,
            'detector_top1_accuracy_test': 0
        }
    else:
        logs3 = algo.update_detector_parameters(exps)
        algo.update_detector_beliefs(exps)

    logs2 = algo.update_ac_parameters(exps)
    

    logs = {**logs1, **logs2, **logs3}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)

        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"], logs["num_frames_per_episode"])
        average_discounted_return = utils.average_discounted_return(logs["return_per_episode"], logs["num_frames_per_episode"], args.discount)
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["average_reward_per_step", "average_discounted_return"]
        data += [average_reward_per_step, average_discounted_return]
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        header += ["detector_loss_train", "detector_loss_test", "detector_top1_accuracy_train", "detector_top1_accuracy_test"]
        data += [logs["detector_loss_train"], logs["detector_loss_test"], logs["detector_top1_accuracy_train"], logs["detector_top1_accuracy_test"]]

        txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | ARPS: {:.3f} | ADR: {:.3f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | dLtrain {:.3f} | dLtest {:.3f} | dtop1Acctrain {:.3f} | dtop1Acctest {:.3f}"
            .format(*data))
        header += ["detector_grad_train", "detector_grad_test"]
        data += [logs["detector_grad_train"], logs["detector_grad_test"]]
        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            wandb.log({field: value})

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": algo.acmodel.state_dict(),
                  "detector_model_state": algo.detectormodel.state_dict(),
                  "optimizer_state": algo.optimizer.state_dict()}
        if algo.optimizer_detector is not None:
            status["optimizer_detector_state"] = algo.optimizer_detector.state_dict()
        if hasattr(preprocess_obss, "vocab") and preprocess_obss.vocab is not None:
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir + "/train")
        txt_logger.info("Status saved")