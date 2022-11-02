#!/usr/bin/env python3

import sys
import os
from pprint import pformat
from pprint import pprint

sys.path.append(os.path.join(os.getcwd()))

from utils.des_bench import DesignBenchFunctionWrapper

import logging
import torch
import math
import argparse
import pickle as pkl
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from mingpt.model import GPT, GPTConfig
from mingpt.model_discrete_new import GPTDiscrete
from mingpt.model_discrete_new import GPTConfig as GPTConfigDiscrete

from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import set_seed

from utils.utils import LogResult

from contextlib import contextmanager, redirect_stderr, redirect_stdout

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[0])
parser.add_argument('--context_length', type=int, default=40)
# parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cond_rtgs', type=float, nargs="+", default=[8.0])
parser.add_argument('--data_dir_prefix', type=str, default='./generated_datasets/')
parser.add_argument('--dataset', type=str, default='dkitty_800x128_sorted_64.p')
parser.add_argument('--eval_dataset', type=str, default='dkitty/dkitty_sorted_128x128_64_eval.p')
parser.add_argument('--experiment', type=str, default='test')
# parser.add_argument('--train', action='store_true')
# parser.add_argument('--resume', action='store_true')
# parser.add_argument('--test', action='store_true')
parser.add_argument('--dim', type=int, default=56)
parser.add_argument('--task', type=str, default='dkitty')
parser.add_argument('--no_update_rtg', action='store_true')
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--init_len', type=int, default=64)
parser.add_argument('--vocab_size', type=int, default=1)
parser.add_argument('--surrogate', action='store_true')
parser.add_argument('--suffix', type=str, default="")
parser.add_argument('--layers', type=int, default=16)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--max_timestep', type=int, default=128)
parser.add_argument('--model_stage', type=str, default="last")

args = parser.parse_args()


dim = args.dim
optima = 1.0

# setup logging
log_filename = f'{args.experiment}_eval'
if (args.suffix != ""):
    log_filename += f"_{args.suffix}"
log_filename += ".log"
if not os.path.exists(f"logs/{args.task}/{args.experiment}/"):
    os.makedirs(f"logs/{args.task}/{args.experiment}/")
logging.basicConfig(
        filename=f"logs/{args.task}/{args.experiment}/{log_filename}",
        level=logging.INFO,
        datefmt="%Y/%m/%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",)

logging.info("-" * 30)

with open(os.path.join(args.data_dir_prefix, args.dataset), 'rb') as f:
    points, values, pointwise_regret, cumulative_regret_to_go, timesteps, _ = pkl.load(f)

with open(os.path.join(args.data_dir_prefix, args.eval_dataset), 'rb') as f:
    points_eval, values_eval, pointwise_regret_eval, cumulative_regret_to_go_eval, timesteps_eval, _ = pkl.load(f)

train_test_split = 9 * (points.shape[0] // 10)

class PointRegretDataset(Dataset):
    def __init__(self, block_size, points, values, pointwise_regret, cumulative_rtg, timesteps):
        self.block_size = block_size
        self.vocab_size = args.vocab_size ### TODO
        self.num_trajectories = points.shape[0]
        self.size_of_trajectory = points.shape[1]
        self.points = points
        self.values = values
        self.pointwise_regret = pointwise_regret
        self.cumulative_rtg = cumulative_rtg
        self.timesteps = timesteps
    
    def __len__(self):
        return self.num_trajectories * self.size_of_trajectory - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 2
        traj_idx = idx // self.size_of_trajectory
        sidx = idx - traj_idx * self.size_of_trajectory
        if sidx + block_size > self.size_of_trajectory:
            sidx = self.size_of_trajectory - block_size

        eidx = sidx + block_size

        points = self.points[traj_idx, sidx:eidx] # (block_size, 1)
        values = self.values[traj_idx, sidx:eidx].unsqueeze(-1)
        cumulative_rtgs = self.cumulative_rtg[traj_idx, sidx:eidx].unsqueeze(-1)
        timesteps = self.timesteps[traj_idx, sidx:sidx+1].unsqueeze(-1)

        return points, points, cumulative_rtgs, timesteps

train_dataset = PointRegretDataset(args.context_length * 2, points[:train_test_split], values[:train_test_split], pointwise_regret[:train_test_split], cumulative_regret_to_go[:train_test_split], timesteps[:train_test_split])
test_dataset = PointRegretDataset(args.context_length * 2, points[train_test_split:], values[train_test_split:], pointwise_regret[train_test_split:], cumulative_regret_to_go[train_test_split:], timesteps[train_test_split:])

print(points_eval.shape, values_eval.shape)
ini_len = args.init_len
logging.info(f"Initialisation length: {ini_len}")



# print("pr", init_prs)
# print("crtg", init_crtg)

# print(initial_points.shape, initial_rtgs.shape)


use_oracle = True
if (args.surrogate):
    use_oracle = False

func_vals = [[-1 for j in args.seeds] for i in args.cond_rtgs]
oracle_func_vals = [[-1 for j in args.seeds] for i in args.cond_rtgs]
for j, seed in enumerate(args.seeds):

    set_seed(seed)

    print("Initialisation length is", args.init_len)
    # TODO: change from hardcoded value
    max_timestep = args.max_timestep

    if args.discrete:
        print("Loading discrete model")
        mconf = GPTConfigDiscrete(train_dataset.vocab_size, train_dataset.block_size, input_dim=dim,
                    n_layer=args.layers, n_head=args.heads, n_embd=128, max_timestep=max_timestep)
        model = GPTDiscrete(mconf)

    else:
        mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, input_dim=dim,
                    n_layer=args.layers, n_head=args.heads, n_embd=128, max_timestep=max_timestep)
        model = GPT(mconf)


    #load model
    # model.load_state_dict(torch.load('checkpoints/{}/{}/{}_{}_best'.format(args.task, args.experiment, args.experiment, seed)))
    if args.model_stage == "last":
        model.load_state_dict(torch.load('checkpoints/{}/{}/{}_{}'.format(args.task, args.experiment, args.experiment, seed)))
    else:
        model.load_state_dict(torch.load('checkpoints/{}/{}/{}_{}_best'.format(args.task, args.experiment, args.experiment, seed)))

    model = model.cuda()

    # produce results
    tconf = None
    update_rtg = True
    if (args.no_update_rtg):
        update_rtg = False

    func = DesignBenchFunctionWrapper(args.task, normalise=True, oracle=use_oracle)
    print(func.optima, func.min, func.max)
    for i, cond_rtg in enumerate(args.cond_rtgs):

        if ini_len > 0:
            ind = np.random.randint(0, 128)
            #ind = 0
            print("index: ", ind)
            initial_points = points_eval[ind, 0:ini_len, :]
            init_prs = pointwise_regret_eval[ind, :ini_len]
            initial_rtgs = torch.flip(torch.cumsum(torch.flip(init_prs, dims=[0]), 0), dims=[0])
            # print(init_prs)
            # print(initial_rtgs)
        else:
            initial_points = None
            initial_rtgs = None
        print("Evaluating for expt", args.experiment, "for seed", seed, "cond_rtg", cond_rtg, "update_rtg", update_rtg)
        logging.info(f"Evaluating for expt {args.experiment}, for seed {seed}, cond_rtg {cond_rtg}, update_rtg {update_rtg}")

        results = LogResult(exptname=args.task + f"/{args.experiment}/{cond_rtg}_" + str(update_rtg) + "_" + str(args.init_len) + "_" + args.model_stage)
        res = {}

        # with suppress_output():
        res['points'], res['simple_regret'] = model.evaluate(rtg=cond_rtg,
                                                                unroll_length=max_timestep,
                                                                function=func,
                                                                device='cuda', 
                                                                update_regret=update_rtg,
                                                                initial_points=initial_points,
                                                                initial_rtgs=initial_rtgs)
                                                                
        assert 'points' in res
        #TODO print results
        res_arr = np.array(res['simple_regret'])
        min_reg = np.min(res_arr)
        min_reg_idx = np.argmin(res_arr)
        func_value_normalized = func.optima - min_reg
        func_value = func_value_normalized * (func.max - func.min) + func.min
        print("Function value of best point:", func_value)
        logging.info(f"Function value of best point: {func_value}")
        func_vals[i][j] = func_value
        oracle_func_vals[i][j] = func_value
        if (args.surrogate):
            qq = np.asarray(res['points'])
            # ground_truth_predictions = func.task.predict(qq)
            ground_truth_predictions = func.task.predict(qq[min_reg_idx].reshape(1, -1))
            print(f"Ground truth best point: {ground_truth_predictions.max()}")
            logging.info(pformat(f"Ground truth best point: {ground_truth_predictions.max()}"))
            oracle_func_vals[i][j] = ground_truth_predictions
        results.save(points=res['points'], instantaneous_regret=res['simple_regret'], seed=seed, conditioned_rtg=cond_rtg, model_config=mconf, train_config=tconf)

# print(func_vals)
means = []
stds = []
gt_means = []
gt_stds = []
for i, cond_rtg in enumerate(args.cond_rtgs):

    mean = round(np.mean(np.array(func_vals[i])), 2)
    std = round(np.std(np.array(func_vals[i])), 2)
    gt_mean = round(np.mean(np.array(oracle_func_vals[i])), 2)
    gt_std = round(np.std(np.array(oracle_func_vals[i])), 2)

    print("cond_rtg", cond_rtg, ":", mean, "±", std)
    logging.info(f"cond_rtg: {cond_rtg}: {mean} ± {std}")
    log = f"GT cond_rtg: {cond_rtg}: {gt_mean} ± {gt_std}"
    print(log)
    logging.info(log)
    means.append(mean)
    stds.append(std)
    gt_means.append(gt_mean)
    gt_stds.append(gt_std)

print("cond_rtgs", args.cond_rtgs)
logging.info(f"Conditioning RTG: {pformat(args.cond_rtgs)}")
print("means", means)
logging.info(f"Means: {means}")
print("stds", stds)
logging.info(f"Standard deviations: {stds}")
print("gt means", gt_means)
logging.info(f" GT Means: {gt_means}")
print("gt stds", gt_stds)
logging.info(f"GT Standard deviations: {gt_stds}")

best_regret_idx = np.argmax(means)
log = f"Best regret: {means[best_regret_idx]}, Best RTG: {args.cond_rtgs[best_regret_idx]}"
print(log)
logging.info(log)
# Best eval method
func_vals = np.asarray(func_vals)
values = []
for i, seed in enumerate(args.seeds):
    values.append(np.max(func_vals[:,i]))
    logging.info(pformat(func_vals[:,i]))
    top2 = func_vals[:, i].argsort()[-2:]
    log = f"For seed {seed}: top 2: {top2}"
    print(log)
    logging.info(log)

values = np.asarray(values)
logging.info(pformat(values))
log = f"Optimising over RTG; Mean: {values.mean()} Std: {values.std()}"
print(log)
logging.info(log)

logging.info("-" * 30)

logging.shutdown()
