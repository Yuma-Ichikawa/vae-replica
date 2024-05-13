import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from src import utils
from src import data
from src import vae

def replica_reproduce_1rank(N, alpha_list, num_seed, device="cpu", rho=1., eta=1., num_epoch=10000, beta=1.0, lr=0.001, reg_param=1.0):

    P_list = N*np.array(alpha_list)
    seed_list=[i*100 for i in range(num_seed)]
    W0=torch.ones(N, 1)

    results = []
    for P in P_list:
        alpha_results = []
        print(f"alpha: {P/N:.2f}")
        for seed in seed_list:
            utils.fix_seed(seed)
            dataset=data.make_dataset_from_SCM(P=int(P), N=int(N), M=1, W0=W0, eta=1.0, rho=1.0).to(device)
            model = copy.deepcopy(vae.LinearVAE(N, 1).to(device))
            history, model = vae.batchfit_linearvae(dataset, model, num_epoch=num_epoch, beta=beta, lr=lr, reg_param=reg_param, device=device, check_interval=500)
            alpha_results.append([history["elbo"][-1],
                                history["m"][-1],
                                history["d"][-1],
                                history["Q"][-1],
                                history["E"][-1],
                                history["R"][-1],
                                history["D"][-1],
                                history["eg_order"][-1]])
        results.append(alpha_results)
    return results