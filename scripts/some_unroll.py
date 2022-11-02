import pickle as pkl 
import os, sys 

sys.path.append(os.path.join(os.getcwd()))

# from utils.des_bench import DesignBenchFunctionWrapper
import mingpt
import numpy as np 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--seeds', type=int, nargs='+', default=[123, 234, 345, 456, 567])
parser.add_argument('--experiment', type=str, default='sorted_with_32_layers')
parser.add_argument('--task', type=str, default='ant')
parser.add_argument('--init_lens', type=int, nargs='+', default=[0, 64, 96])
parser.add_argument('--cond_rtgs', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.1, 0.5])
parser.add_argument('--unroll_len', type=int, default=64)
parser.add_argument('--update_rtg', type=bool, default=False)
parser.add_argument('--k', type=int, default=0)


args = parser.parse_args()

BOUNDS = {
        "tf-bind-10": [-1.8585268259048462, 2.128706693649292]       
 }

# cond_rtgs = [0.0, 0.01, 0.05, 1.0, 2.0, 4.0, 6.0, 8.0]
#cond_rtgs = [0.0, 0.01, 0.05, 0.1, 0.5]
cond_rtgs = args.cond_rtgs
seeds = args.seeds
update_rtg = args.update_rtg
init_lens = args.init_lens
task = args.task
unroll_len = args.unroll_len

expt_name = args.experiment
#expt_name = "sorted_with_64_context_32_layers_17_heads"
#expt_name = "results_sorted_with_32_l_8_h_final"
# func = DesignBenchFunctionWrapper(task, normalise=True)

BOUNDS = {
    "tf-bind-8": [0.0, 1.0],
    "tf-bind-10": [-1.8585268259048462, 2.128706693649292],
    "superconductor": [0.0002099999983329326, 185.0],
    "ant": [-386.9003601074219, 590.2444458007812],
    "dkitty": [-880.45849609375, 340.90985107421875],
    "hopper": [87.9349136352539, 1361.610595703125]
 }

bounds = BOUNDS[task]
mini, maxi = bounds[0], bounds[1]
optima = 1.0

print("update_rtg", update_rtg)

for init_len in init_lens:
    print("="*20, "init length", init_len, "="*20)
    max_vals = []
    for seed in seeds:
        rtg_vals = []
        for rtg in cond_rtgs:

            filename = "results/" + task + f"/" + expt_name + f"/{args.k}_" + str(rtg) + "_" + str(update_rtg) + "_" + str(init_len) + "_last/" + str(seed)
            # print(filename)
            res = pkl.load(open(filename, 'rb'))
            print("ppp", len(res['regret']))
            # print(res['regret'])
            # exit(0)
            temp_res = res['regret'][0:args.unroll_len]
            min_reg = np.min(np.array(temp_res))
            func_value_normalized = optima - min_reg
            if task == "hopper":
                func_value = func_value_normalized
            else:
                func_value = func_value_normalized * (maxi - mini) + mini
            # print("Function value of best point:", func_value)
            print("rtg_wise", seed, rtg, func_value)
            rtg_vals.append(func_value)
        # print(rtg_vals)
        max_val = max(rtg_vals)
        max_ind = rtg_vals.index(max_val)
        print("seed", seed, "max func val", max_val, "occured at rtg", cond_rtgs[max_ind])
        max_vals.append(max_val)
    print("Final result: ", np.mean(np.array(max_vals)), "+-", np.std(np.array(max_vals)))
        

