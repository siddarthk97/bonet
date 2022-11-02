#!/usr/bin/env python3

import sys
import os
from pprint import pformat
from contextlib import contextmanager, redirect_stderr, redirect_stdout

sys.path.append(os.path.join(os.getcwd()))

from utils.des_bench import DesignBenchFunctionWrapper
@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset

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
from utils.utils import class_to_dict

TASKNAME2FULL = {
        'dkitty': DKittyMorphologyDataset,
        'ant': AntMorphologyDataset,
        'tf-bind-8': TFBind8Dataset,
        'tf-bind-10': TFBind10Dataset,
        'superconductor': SuperconductorDataset,
        'nas': CIFARNASDataset,
        }

TASKNAME2TASK = {
        'dkitty': 'DKittyMorphology-Exact-v0',
        'ant': 'AntMorphology-Exact-v0',
        'tf-bind-8': 'TFBind8-Exact-v0',
        'tf-bind-10': 'TFBind10-Exact-v0',
        'superconductor': 'Superconductor-RandomForest-v0',
        'nas': 'CIFARNAS-Exact-v0',
        }

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--context_length', type=int, default=40)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--cond_rtgs', nargs="+", default=[8])
parser.add_argument('--data_dir_prefix', type=str, default='./generated_datasets/')
parser.add_argument('--dataset', type=str, default='dkitty_800x128_sorted_64.p')
parser.add_argument('--experiment', type=str, default='test')
# parser.add_argument('--train', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--sampled', action='store_true')
# parser.add_argument('--test', action='store_true')
parser.add_argument('--dim', type=int, default=56)
parser.add_argument('--task', type=str, default='dkitty')
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--vocab_size', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--heads', type=int, default=16)
parser.add_argument('--max_timestep', type=int, default=128)
parser.add_argument('--add_noise', action='store_true')
# parser.add_argument('--no_update_rtg', action='store_true')

args = parser.parse_args()

set_seed(args.seed)

dim = args.dim
optima = 1

# setup logging
log_filename = f'{args.experiment}_{args.seed}_train.log'
if not os.path.exists(f"logs/{args.task}/{args.experiment}/"):
    os.makedirs(f"logs/{args.task}/{args.experiment}/")
logging.basicConfig(
        filename=f"logs/{args.task}/{args.experiment}/{log_filename}",
        level=logging.INFO,
        datefmt="%Y/%m/%d %H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",)

logging.info('-' * 30)
logging.info(f"Starting training run for seed {args.seed}")

with open(os.path.join(args.data_dir_prefix, args.dataset), 'rb') as f:
    points, values, pointwise_regret, cumulative_regret_to_go, timesteps, _ = pkl.load(f)
print("dddddd", points.device)
train_test_split = 9 * (points.shape[0] // 10)

class PointRegretDataset(Dataset):
    def __init__(self, block_size, points, values, pointwise_regret, cumulative_rtg, timesteps, add_noise=False, variance=0.01):
        self.block_size = block_size
        self.vocab_size = args.vocab_size   # TODO
        self.num_trajectories = points.shape[0]
        self.size_of_trajectory = points.shape[1]
        self.points = points
        self.values = values
        self.pointwise_regret = pointwise_regret
        self.cumulative_rtg = cumulative_rtg
        self.timesteps = timesteps

        self.add_noise = add_noise
        self.noise = torch.rand(self.num_trajectories, 1).repeat(1, self.size_of_trajectory) * variance

        # if self.add_noise:
        #     print("added noise: ", self.noise)

    
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
        
        # values = values.unsqueeze(-1)
        cumulative_rtgs = self.cumulative_rtg[traj_idx, sidx:eidx]

        # if self.add_noise:
        #     cumulative_rtgs += self.noise[traj_idx, sidx:eidx]
            
        cumulative_rtgs = cumulative_rtgs.unsqueeze(-1)
        timesteps = self.timesteps[traj_idx, sidx:sidx+1].unsqueeze(-1)

        return points, points, cumulative_rtgs, timesteps

class SamplingDataset(Dataset):
    def __init__(self, block_size, taskname, num_trajectories, size_of_trajectory, num_bins):
        self.block_size = block_size
        self.vocab_size = 1
        self.num_trajectories = num_trajectories
        self.size_of_trajectory = size_of_trajectory
        self.num_bins = num_bins
        self.taskname = taskname
        self.task = design_bench.make(TASKNAME2TASK[self.taskname])
        self.full_task = TASKNAME2FULL[self.taskname]()
        self.x = self.task.x
        self.y = (self.task.y - self.full_task.y.min()) / (self.full_task.y.max() - self.full_task.y.min())
        self.optima = 1.
        self.regrets = self.optima - self.y
        self.regrets = torch.tensor(self.regrets)
        self.inds = torch.argsort(-self.regrets)
        print("Sorted regrets:", self.regrets[self.inds])
        print("Unique points", torch.unique(points))

        min_reg = torch.min(self.regrets)
        max_reg = torch.max(self.regrets)

        bin_len = (max_reg - min_reg) / self.num_bins
        print(min_reg, max_reg, bin_len)

        self.bins = [[] for i in range(self.num_bins)]

        for i in range(len(self.y)):
            # find the bin
            for b in range(self.num_bins):
                if self.regrets[i] >= min_reg + b * bin_len and self.regrets[i] <= min_reg + (b + 1) * bin_len:
                    self.bins[b].append(i)
                    break

        self.nis = [len(i) for i in self.bins]
        self.exps = [-1 for i in self.bins]
        self.scores = [-1 for i in self.bins]

        self.tau = self.optima - np.percentile(self.regrets, 90)
        self.K = 0.03 * self.num_trajectories
        print("tau: ", self.tau, " K: ", self.K)

        for b in range(len(self.bins)):
            low = self.optima - (min_reg + b * bin_len)
            high = self.optima - (min_reg + (b + 1) * bin_len)
            avg = (low + high) / 2
            self.exps[b] = np.exp((avg - self.optima) / self.tau)

        for b in range(len(self.bins)):
            self.scores[b] = (self.nis[b] / (self.nis[b] + self.K)) * self.exps[b]

        self.scores = np.array(self.scores)
        self.scores = self.size_of_trajectory * (self.scores / np.sum(self.scores))
        self.scores = np.round(self.scores).astype(int)
        self.scores[0] += (self.size_of_trajectory - np.sum(self.scores))

        assert np.sum(self.scores) == self.size_of_trajectory
        print("counts: ", self.nis)
        # print(exps)
        print("scores", self.scores)
        self.dataset = torch.cat([torch.tensor(self.x), self.regrets.reshape(-1, 1)], dim=1)

    def __len__(self):
        return self.num_trajectories * self.size_of_trajectory - self.block_size

    def __getitem__(self, idx):
        # create trajectory
        nums_each = self.size_of_trajectory // self.num_bins
        lst = []
        val_lst = []

        for b in range(self.num_bins):
            inds = list(np.random.choice(self.bins[b], self.scores[b], replace=True))
            subs = self.dataset[inds, :-1]
            regs = self.dataset[inds, -1]
            vals = self.optima - regs
            val_lst.append(vals)
            lst.append(subs)

        trajectory = torch.cat(lst)
        vals = torch.cat(val_lst)
        trajectory = torch.flip(trajectory, dims=(0, ))
        # print("shapee: ", trajectory.shape, vals.shape)
        vals = torch.flip(vals, dims=(0, ))
        indxs = torch.argsort(vals)
        vals = vals[indxs]
        trajectory = trajectory[indxs, :]
        trajectory = trajectory.unsqueeze(0)
        vals = vals.unsqueeze(0)

        pr = self.optima - vals
        cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pr, [1]), 1), [1])
        timesteps = torch.arange(self.size_of_trajectory).unsqueeze(0)

        # trim to start and end indices
        block_size = self.block_size // 2
        # traj_idx = idx // self.size_of_trajectory
        sidx = idx % self.size_of_trajectory
        # sidx = idx - traj_idx * self.size_of_trajectory
        if sidx + block_size > self.size_of_trajectory:
            sidx = self.size_of_trajectory - block_size

        eidx = sidx + block_size

        points = trajectory[0, sidx:eidx] # (block_size, 1)
        values = vals[0, sidx:eidx].unsqueeze(-1)
        cumulative_rtgs = cumulative_regret_to_go[0, sidx:eidx].unsqueeze(-1)
        timesteps = timesteps[0, sidx:sidx+1].unsqueeze(-1)

        return points, points, cumulative_rtgs, timesteps

if (not args.sampled):
    train_dataset = PointRegretDataset(args.context_length * 2, points[:train_test_split], values[:train_test_split], pointwise_regret[:train_test_split], cumulative_regret_to_go[:train_test_split], timesteps[:train_test_split], add_noise=args.add_noise)
else:
    # TODO: remove hardcoded values
    train_dataset = SamplingDataset(args.context_length * 2, args.task, 800, 128, num_bins=64)
test_dataset = PointRegretDataset(args.context_length * 2, points[train_test_split:], values[train_test_split:], pointwise_regret[train_test_split:], cumulative_regret_to_go[train_test_split:], timesteps[train_test_split:])

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

logging.info("================ MODEL CONFIG ================")
logging.info(pformat(class_to_dict(mconf)))
logging.info("==============================================")

if not os.path.exists(os.path.join('checkpoints/' + args.task, args.experiment)):
    os.makedirs(os.path.join('checkpoints/' + args.task, args.experiment))

if (args.resume):
    model.load_state_dict(torch.load('checkpoints/{}/{}/{}_{}'.format(args.task, args.experiment, args.experiment, args.seed)))
    model = model.cuda()

writer = SummaryWriter(log_dir='tensorboard/{}_{}_{}'.format(args.task, args.experiment, args.seed))
tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                            lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*2,
                            num_workers=4, seed=args.seed, max_timestep=max_timestep,
                            ckpt_path='checkpoints/{}/{}/{}_{}'.format(args.task, args.experiment, args.experiment, args.seed))
trainer = Trainer(model, train_dataset, test_dataset, tconf, add_noise=args.add_noise)

logging.info("================ TRAIN CONFIG ================")
logging.info(pformat(class_to_dict(tconf)))
logging.info("==============================================")

trainer.train(writer=writer)
writer.close()
logging.info('-' * 30)

logging.shutdown()
