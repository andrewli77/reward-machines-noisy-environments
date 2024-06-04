import argparse
import wandb
import sys
import utils
import os
from detector_model import *
from algo_pretrain_detector import Algo 

import torch
import numpy
from torch_ac.utils import DictList
import torch.nn.functional as F

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--train-data-path", default=None,
                        help="Path of collected train data")
    parser.add_argument("--test-data-path", default=None,
                        help="Path of collected test data")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Log the experiment with weights & biases")

    ## Parameters for main algorithm
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="number of epochs for train[ing the detector model")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="batch size (default: 50)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--rm-update-algo", type=str, default="tdm",
                        help="[tdm, naive, ibu, oracle]")
    parser.add_argument("--use-mem", action="store_true", default=False,
                        help="whether the model is recurrent")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set run dir

    default_model_name = f"{args.env}_{args.rm_update_algo}_pretrain_seed{args.seed}"
    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)

    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    if not args.wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb.init(project='noisy-detector')
    wandb.run.name = default_model_name
    wandb.run.save()
    config = wandb.config
    config.update(args)

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    txt_logger.info(f"Device: {device}\n")

    # Load training status
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"epochs": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load env
    env = utils.make_env(args.env, "oracle", args.seed)
    obs_space, _ = utils.get_obss_preprocessor(env)
    txt_logger.info("Environments loaded\n")

    # Load representation model
    detector_model = getDetectorModel(env, obs_space, args.rm_update_algo, args.use_mem).to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(detector_model))
    if "model_state" in status:
        detector_model.load_state_dict(status["model_state"])
        print("Loaded state from existing run.")
    best_test_loss = 10000000000000.
    if "best_test_loss" in status:
        best_test_loss = status["best_test_loss"]
        print("Best test loss:", best_test_loss)

    # Load data
    train_data = torch.load(args.train_data_path)
    test_data = torch.load(args.test_data_path)
    n_episodes = train_data['obss'].shape[0]
    assert(args.batch_size <= n_episodes)


    train_exps = DictList()
    train_exps.obs = train_data['obss'].transpose(0,1).to(device)
    train_exps.events = train_data['events'].transpose(0,1).to(device)
    train_exps.rm_states = train_data['rm_states'].transpose(0,1).to(device)
    train_exps.mask = train_data['masks'].transpose(0,1).to(device)

    # Load algo
    #lr = hparams[args.seed-1][1]
    algo = Algo(env, test_data, detector_model, device, args.optim_eps, args.lr, args.rm_update_algo, args.use_mem)

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])


    # Train model
    epochs = status["epochs"]
    update = status["update"]

    while epochs < args.epochs:
        # Update model parameters

        shuffled = numpy.arange(0, n_episodes)
        shuffled = numpy.random.permutation(shuffled)
        batch_size = args.batch_size

        running_event_loss = 0.
        running_rm_state_loss = 0.
        running_grad_norm = 0.

        for batch in range(n_episodes // batch_size):
            indexes = shuffled[batch * batch_size: (batch+1) * batch_size]

            logs = algo.update_parameters(train_exps[:,indexes])

            running_event_loss += logs['event_loss']
            running_rm_state_loss += logs['rm_state_loss']
            running_grad_norm += logs['grad_norm']


        # Logging

        all_headers = ['event_loss', 'rm_state_loss', 'grad_norm']

        header = []
        data = []

        for h in all_headers:
            header += ["train_"+h]
        data += [running_event_loss / (n_episodes // batch_size), running_rm_state_loss / (n_episodes // batch_size), running_grad_norm / (n_episodes // batch_size)]


        if update % args.log_interval == 0:
            
            evaluate_logs = algo.evaluate()

            for h in all_headers:
                if h in evaluate_logs:
                    header += ["test_"+h]
                    data += [evaluate_logs[h]]

            if args.rm_update_algo in ["tdm"]:
                if evaluate_logs['rm_state_loss'] < best_test_loss:
                    best_test_loss = evaluate_logs['rm_state_loss']
                    status_best = {
                        "model_state": algo.detector_model.state_dict()
                    }
                    utils.save_status(status_best, model_dir + "/best")
            elif args.rm_update_algo in ["naive", "ibu"]:
                if evaluate_logs['event_loss'] < best_test_loss:
                    best_test_loss = evaluate_logs['event_loss']
                    status_best = {
                        "model_state": algo.detector_model.state_dict()
                    }
                    utils.save_status(status_best, model_dir + "/best")


        if status["epochs"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            wandb.log({field: value})

        # Save status
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"epochs": epochs,
                      "update": update,
                      "best_test_loss": best_test_loss,
                      "model_state": algo.detector_model.state_dict(),
                      "optimizer_state": algo.optimizer.state_dict(),
                      }
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        print(epochs, "epochs")
        epochs += 1
        update += 1
