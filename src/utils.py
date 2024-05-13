import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from src import vae

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calc_observable(model, W0=None, device="cpu"):

    input_dim = model.dec.weight.size(0)
    latent_dim = model.dec.weight.size(1)

    if W0==None:
        W0=torch.ones(model.dec.weight.size(0), model.dec.weight.size(1), device=device)
        m = (model.dec.weight.T @ W0)/input_dim
        d = (model.mu.weight @ W0)/input_dim

    else:
        m = (model.dec.weight.T @ W0)/input_dim
        d = (model.mu.weight @ W0)/input_dim

    Q = (model.dec.weight.T @ model.dec.weight)/input_dim
    E = (model.mu.weight @ model.mu.weight.T)/input_dim
    R = (model.dec.weight.T @ model.mu.weight.T)/input_dim
    D = model.var


    if latent_dim==1:

        m = m.flatten()[0]
        d = d.flatten()[0]
        Q = Q.flatten()[0]
        E = E.flatten()[0]
        R = R.flatten()[0]
        # Consider solutions with inversion symmetry
        eg_order = 1-(2*torch.abs(m))+Q

        ob_list = [
                m.item(),
                d.item(),
                Q.item(),
                E.item(),
                R.item(),
                D.item(),
                eg_order.item()
                ]
    else:
        # Consider solutions with inversion symmetry
        eg = 1-2*torch.abs(torch.sum(m))+torch.sum(q)

        ob_list = [
                m.flatten().tolist(),
                d.flatten().tolist(),
                Q.flatten().tolist(),
                E.flatten().tolist(),
                R.flatten().tolist(),
                D.tolist(),
                eg_order.item()
                ]

    return ob_list