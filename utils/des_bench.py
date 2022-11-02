import sys
import os
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

with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset

import numpy
import torch

from .forward import ForwardModel, ProbabilisticForwardModel

TASKNAME2TASK = {
        'dkitty': 'DKittyMorphology-Exact-v0',
        'ant': 'AntMorphology-Exact-v0',
        'tf-bind-8': 'TFBind8-Exact-v0',
        'tf-bind-10': 'TFBind10-Exact-v0',
        'superconductor': 'Superconductor-RandomForest-v0',
        'nas': 'CIFARNAS-Exact-v0',
        'chembl': 'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0',
        }

class DesignBenchFunctionWrapper:
    def __init__(self, taskname, normalise=False, optima=1, oracle=True):
        self.optima = optima
        self.taskname = taskname
        self.task = design_bench.make(TASKNAME2TASK[self.taskname])

        self.oracle = oracle
        if (not self.oracle):
            self.forward_net = ForwardModel(hidden_size=128, input_size=self.task.x.shape[-1])
            # self.forward_net = ProbabilisticForwardModel(hidden_size=128, input_size=self.task.x.shape[-1])
            self.forward_net = torch.nn.DataParallel(self.forward_net).to('cuda')
            self.forward_net.load_state_dict(torch.load(f"forward_checkpoints/{taskname}_best"))
            # self.forward_net.load_state_dict(torch.load(f"forward_checkpoints/probabilistic_{taskname}_best"))

        self.max = None
        self.min = None
        self.normalise = normalise
        if (normalise):
            # override optima to be 1 if normalised
            self.optima = 1
            # self.task.map_normalize_y()
            fully_observed_task = None
            if self.taskname == 'tf-bind-8':
                fully_observed_task = TFBind8Dataset()
            elif self.taskname == 'tf-bind-10':
                fully_observed_task = TFBind10Dataset()
            elif self.taskname == 'dkitty':
                fully_observed_task = DKittyMorphologyDataset()
            elif self.taskname == 'ant':
                fully_observed_task = AntMorphologyDataset()
            elif self.taskname == 'superconductor':
                fully_observed_task = SuperconductorDataset()
            elif self.taskname == 'nas':
                fully_observed_task = CIFARNASDataset()
            elif self.taskname == 'chembl':
                assay_chembl_id = 'CHEMBL3885882'
                standard_type = 'MCHC'
                fully_observed_task = ChEMBLDataset(assay_chembl_id=assay_chembl_id, standard_type=standard_type)
            else:
                raise NotImplementedError()

            self.max = fully_observed_task.y.max()
            self.min = fully_observed_task.y.min()

            print("=" * 20)
            print("Task name:", self.taskname, "optima:", self.optima,  "Dataset min/max: {}/{}".format(self.min, self.max))
            print("=" * 20)

    def eval(self, x):
        if self.oracle:
            if torch.is_tensor(x):
                x = x.view(1, -1)
                y = self.task.predict(x.cpu().numpy())
            else:
                y = self.task.predict(x)
        else:
            if torch.is_tensor(x):
                x = x.view(1, -1)
            else:
                x = torch.tensor(x, dtype=torch.float32)

            with torch.no_grad():
                y = self.forward_net(x.to('cuda'))
                # y = y[:,0]

            y = y * (self.max - self.min) + self.min

        if self.normalise:
            assert self.max is not None
            assert self.min is not None
            y = (y - self.min) / (self.max - self.min)
        return float(y)

    def eval_unnormalise(self, x):
        if torch.is_tensor(x):
            x = x.view(1, -1)
            y = self.task.predict(x.cpu().numpy())
        else:
            y = self.task.predict(x)

    def regret(self, x):
        return self.optima - self.eval(x)

    def reward(self, x):
        raise NotImplementedError

if __name__ == "__main__":
    import pickle as pkl 

    points, _, _, _, _, _ = pkl.load(open("../generated_datasets/dkitty/dkitty_sorted_128x128_64_eval.p", "rb"))

    points = points[0, :, :]
    print(points.shape)

    func = DesignBenchFunctionWrapper("dkitty")
    y = func.task.predict(points.cpu().numpy())

    print(y.shape)
    print(y)


