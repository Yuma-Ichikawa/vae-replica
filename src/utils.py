import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from main import vae

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
        M = model.dec.weight.T@torch.ones(model.dec.weight.size(0), model.dec.weight.size(1), device=device)/input_dim
        tM = model.mu.weight @ torch.ones(model.mu.weight.size(1), model.mu.weight.size(0), device=device)/input_dim
    else:
        M = (model.dec.weight.T@W0)/input_dim
        tM = (model.mu.weight@W0)/input_dim

    Q = (model.dec.weight.T @ model.dec.weight)/input_dim
    tQ = (model.mu.weight @ model.mu.weight.T)/input_dim
    R = (model.dec.weight.T @ model.mu.weight.T)/input_dim
    v = model.var

    if latent_dim==1:
        m = M.flatten()[0]
        tm=tM.flatten()[0]
        q=Q.flatten()[0]
        tq=tQ.flatten()[0]
        r = R.flatten()[0]
        eg = 1-(2*torch.abs(m))+q
        ob_list = [m.item(), tm.item(), q.item(), tq.item(), r.item(), v.item(), eg.item()]
    else:
        eg = 1-2*torch.sum(M)+torch.sum(Q)
        ob_list = [M.flatten().tolist(), tM.flatten().tolist(), Q.flatten().tolist(), tQ.flatten().tolist(), R.flatten().tolist(), v.tolist(), eg.item()]

    return ob_list