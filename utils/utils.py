import logging
import torch
import os

import pickle as pkl
import numpy as np
from attrdict import AttrDict

# from .plot import plot_simple_regret

def cumargmin(array):
    assert len(array.shape) == 1

    ret = []
    m = float('inf')
    m_idx = -1
    for i in range(array.shape[0]):
        if array[i] < m:
            m = array[i]
            m_idx = i
        ret.append(m_idx)

    return np.asarray(ret)

def class_to_dict(obj):
    attributes = [x for x in dir(obj) if not x.startswith('__')]

    ret = dict()
    for attr in attributes:
        ret[attr] = getattr(obj, attr)

    return ret

class LogResult:
    def __init__(self, exptname, basedir='./results'):
        self._exptname = exptname
        self._basedir = basedir

        self._logfolder = os.path.join(self._basedir, self._exptname)
        if not os.path.exists(self._logfolder):
            os.makedirs(self._logfolder)

    def save(self, points, instantaneous_regret, seed, **params):
        results = AttrDict()
        results['points'] = points
        results['regret'] = instantaneous_regret
        results['seed'] = seed
        for k, v in params.items():
            results[k] = v

        # filename = f"{self._exptname}_{seed}"
        filename = f"{seed}"
        with open(os.path.join(self._logfolder, filename), 'wb') as f:
            pkl.dump(results, f)

    def aggregate_results(self):
        files = os.listdir(self._logfolder)
        print("Caluculating average for ", len(files), " seeds")
        results = []
        for file in files:
            fullpath = os.path.join(self._logfolder, file)
            with open(fullpath, 'rb') as f:
                results.append(pkl.load(f))

        return results

    def compute_statistics(self, task_max, task_min):
        """
        Returns mean and std of results
        """
        results = self.aggregate_results()
        x = []
        for r in results:
            x.append([(1-regret) * (task_max - task_min) + task_min for regret in r.regret])

        x = np.asarray(x)
        x = np.max(x, axis=1)
        return np.mean(x), np.std(x)

class FunctionWrapper:
    def __init__(self, optima, functional, minimum, maximum):
        ### minimum maximum are hidden dataset minimum maximum, optima might be normalized optima
        self.optima = optima
        self.min = minimum
        self.max = maximum
        self.functional = functional

    def eval(self, x):
        return self.functional(x)

    def regret(self, x):
        e = self.eval(x)
        e = (e - self.min) / (self.max - self.min)
        r = self.optima - e
        return r

    def reward(self, x):
        r = self.optima - self.eval(x)
        return 100.0 / (1 + r)

class GPFunctionWrapper:
    def __init__(self, x, y):
        # x: points at which GP has been sampled
        # y: fn evals at points x
        self._x = x.view(-1)
        self._y = y.view(-1)
        self.optima = self._y.max()

    def eval(self, x):
        # Find the point in _x closest to the target point x
        x = x.to(self._x.device)
        diff = x - self._x
        diff = torch.abs(diff)

        idx = torch.argmin(diff)
        return self._y[idx]

    def regret(self, x):
        return self.optima - self.eval(x)

    def reward(self, x):
        raise NotImplementedError
