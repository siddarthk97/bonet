#! /usr/bin/env python3

import sys
import os

sys.path.append(os.path.join(os.getcwd()))


from contextlib import contextmanager, redirect_stderr, redirect_stdout
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
from mingpt.utils import set_seed

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
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset

import torch
import numpy as np 

TASKNAME2FULL = {
        'dkitty': DKittyMorphologyDataset,
        'ant': AntMorphologyDataset,
        'tf-bind-8': TFBind8Dataset,
        'tf-bind-10': TFBind10Dataset,
        'superconductor': SuperconductorDataset,
        'nas': CIFARNASDataset,
        'chembl': ChEMBLDataset,
        }

TASKNAME2TASK = {
        'dkitty': 'DKittyMorphology-Exact-v0',
        'ant': 'AntMorphology-Exact-v0',
        'tf-bind-8': 'TFBind8-Exact-v0',
        'tf-bind-10': 'TFBind10-Exact-v0',
        'superconductor': 'Superconductor-RandomForest-v0',
        'nas': 'CIFARNAS-Exact-v0',
        'chembl': 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0',
        }

# torch.manual_seed(6969)
parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, choices=list(TASKNAME2TASK.keys()), default='dkitty')
parser.add_argument('--seed', type=int, default=6969)
parser.add_argument('--name', type=str, required=True)

args = parser.parse_args()
set_seed(args.seed)
name = args.task

task = design_bench.make(TASKNAME2TASK[args.task])

# data_x = np.load("../bench_dataset_satvik/" + name + "/x.npy")
# data_y = np.load("../bench_dataset_satvik/" + name + "/y.npy")
if args.task == "chembl":
    fully_observed_task = TASKNAME2FULL[args.task](assay_chembl_id="CHEMBL3885882", standard_type="MCHC")
else:
    fully_observed_task = TASKNAME2FULL[args.task]()

data_x = task.x
print("bigger dataset min max", fully_observed_task.y.min(), fully_observed_task.y.max())
print("smaller dataset min max", task.y.min(), task.y.max())

# normalise
data_y = (task.y - fully_observed_task.y.min()) / (fully_observed_task.y.max() - fully_observed_task.y.min())


print("data_x shape", data_x.shape)
data_y = data_y.squeeze(-1)
print("data_y shape", data_y.shape)


data_x = torch.tensor(data_x)
data_y = torch.tensor(data_y)

N = data_x.shape[0]
D = data_x.shape[1]

points = data_x
values = data_y 
# optima = torch.max(values)
optima = 1.0
print("optima in the dataset: ", optima)

regrets = optima - values
print(data_x.shape)
print(data_y.shape) 
print("regrets", regrets)

plt.hist(regrets.detach().numpy(), bins='auto')
plt.savefig("plots/" + args.task + "/regrets_hist.png")

inds = torch.argsort(-regrets)
print("Sorted regrets:", regrets[inds])
print("Unique points", torch.unique(points))

num_bins = 64
traj_len = 128
num_trajectories = 800

min_reg = torch.min(regrets)
max_reg = torch.max(regrets)

bin_len = (max_reg - min_reg) / num_bins
print(min_reg, max_reg, bin_len)

bins = [[] for i in range(num_bins)]

for i in range(len(data_y)):
    # find the bin
    for b in range(num_bins):
        # reg = optima - data_y[i]
        if regrets[i] >= min_reg + b * bin_len and regrets[i] <= min_reg + (b + 1) * bin_len:
            bins[b].append(i)
            break

nis = [len(i) for i in bins]
exps = [-1 for i in bins]
scores = [-1 for i in bins]
print("90th percentile: ", np.percentile(regrets, 90))

tau = optima - np.percentile(regrets, 90)
K = 0.03 * N 
print("tau: ", tau, " K: ", K)

for b in range(len(bins)):
    low = optima - (min_reg + b * bin_len)
    high = optima - (min_reg + (b + 1) * bin_len)
    avg = (low + high) / 2
    exps[b] = np.exp((avg - optima) / tau)
print("exps: ", exps)

for b in range(len(bins)):
    scores[b] = (nis[b] / (nis[b] + K)) * exps[b]

scores = np.array(scores)
scores = traj_len * (scores / np.sum(scores))
# scores = np.floor(scores).astype(int)
scores = np.round(scores).astype(int)
print("Unrounded scores: ", scores, np.sum(scores))
scores[0] += (traj_len - np.sum(scores))

assert np.sum(scores) == traj_len
print("counts: ", nis)
print("scores", scores)

exit(0)

dataset = torch.cat([points, regrets.reshape(-1, 1)], dim=1)

############ train dataset ##############

our_data, our_data_vals = [], []

for i in range(num_trajectories):

    nums_each = traj_len // num_bins
    lst, val_lst = [], []

    for b in range(num_bins):

        inds = list(np.random.choice(bins[b], scores[b], replace=True))
        subs = dataset[inds, :-1]
        regs = dataset[inds, -1]
        vals = optima - regs
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

    our_data.append(trajectory)
    our_data_vals.append(vals)

optima = torch.ones(num_trajectories, 1) * 1.0
print(len(our_data))
our_data = torch.cat(our_data)
our_data_vals = torch.cat(our_data_vals)
pr = optima - our_data_vals

cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pr, [1]), 1), [1])

timesteps = torch.arange(traj_len).repeat(num_trajectories, 1)

obj = [our_data, our_data_vals, pr, cumulative_regret_to_go, timesteps, optima]
pkl.dump(obj, open("generated_datasets/{}/{}".format(args.task, args.name + "_" + str(num_trajectories) + "x" + str(traj_len) + "_" +str(num_bins) + "_train.p"), "wb"))
print("our data shape", our_data.shape)
print("our data vals", our_data_vals.shape)
print("pr shape", pr.shape)
print("cumulative_regret_to_go shape", cumulative_regret_to_go.shape)
print("timesteps shape", timesteps.shape)
# print(timesteps)
print(pr[0, :])
print(cumulative_regret_to_go[0, :])

###################### evaluation trajectories ###################


num_eval_trajectories = 128
our_data_eval = []
our_data_eval_vals = []

for i in range(num_eval_trajectories):

    nums_each = traj_len // num_bins
    lst, val_lst = [], []

    for b in range(num_bins):

        inds = list(np.random.choice(bins[b], scores[b], replace=True))
        subs = dataset[inds, :-1]
        regs = dataset[inds, -1]
        vals = 1.0 - regs ### optima is 1.0
        val_lst.append(vals)
        lst.append(subs)

    trajectory = torch.cat(lst)
    vals = torch.cat(val_lst)
    
    trajectory = torch.flip(trajectory, dims=(0, ))
    vals = torch.flip(vals, dims=(0, ))

    indxs = torch.argsort(vals)
    vals = vals[indxs]
    trajectory = trajectory[indxs, :]

    trajectory = trajectory.unsqueeze(0)
    vals = vals.unsqueeze(0)

    our_data_eval.append(trajectory)
    our_data_eval_vals.append(vals)

optima_eval = torch.ones(num_eval_trajectories, 1) * 1.0
print(len(our_data_eval))
our_data_eval = torch.cat(our_data_eval)
our_data_eval_vals = torch.cat(our_data_eval_vals)
pr = optima_eval - our_data_eval_vals

cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pr, [1]), 1), [1])

timesteps = torch.arange(traj_len).repeat(num_trajectories, 1)

obj = [our_data_eval, our_data_eval_vals, pr, cumulative_regret_to_go, timesteps, optima]
pkl.dump(obj, open("generated_datasets/{}/{}".format(args.task, args.name + "_" + str(num_eval_trajectories) + "x" + str(traj_len) + "_" +str(num_bins) + "_eval.p"), "wb"))
print("our data shape", our_data_eval.shape)
print("our data vals", our_data_eval_vals.shape)
print(pr.shape)
print(cumulative_regret_to_go.shape)
print(timesteps.shape)
# print(timesteps)
print(pr[0, :])
print(cumulative_regret_to_go[0, :])
