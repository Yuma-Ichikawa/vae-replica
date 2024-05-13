import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src import utils
from src import data

class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, W_init=None, tW_init=None):
        super(LinearVAE, self).__init__()

        self.dec = nn.Linear(latent_dim, input_dim, bias=False)
        self.mu = nn.Linear(input_dim, latent_dim, bias=False)
        self.var = nn.Parameter(torch.ones(latent_dim))

        if W_init==None:
            self.dec.weight = nn.Parameter(torch.randn(input_dim, latent_dim))
            self.mu.weight = nn.Parameter(torch.randn(latent_dim, input_dim))
        else:
            self.dec.weight = nn.Parameter(W_init)
            self.mu.weight = nn.Parameter(tW_init)

        self.N = torch.tensor(input_dim)
        self.M = torch.tensor(latent_dim)

    def encode(self, x):

        mu = self.mu(x)/torch.sqrt(self.N)
        var = self.var
        return mu, var

    def decode(self, z):

        hat_x = self.dec(z)/torch.sqrt(self.N)
        return hat_x

    def reparameterize(self, mu, var):

        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def sample(self, n_sample=100, device="cpu"):

        sample_z = torch.randn(n_sample, 
                                self.latent_dim).to(device)
        sample_x = self.decode(sample_z)
        return sample_x

    def forward(self, x):

        mu, var = self.encode(x.view(-1, self.N))
        hat_x = self.decode(mu)
        return hat_x, mu, var

def criterion_for1rank(model, hat_x, x, mu, var, beta=1.0, reg_param=1.0):

    first_term = torch.sum(x**2, dim=1)
    second_term = -2*torch.sum(hat_x*x, dim=1)
    third_term = (torch.sum(model.dec.weight.flatten()**2)/x.size(1))*(torch.sum(mu.pow(2)+var, dim=1))

    each_data_rec_loss = 0.5*(first_term+second_term+third_term)
    recon_loss = torch.sum(each_data_rec_loss)

    # each_data_KLD  = 0.5*torch.sum((-1 - logvar + mu.pow(2) + logvar.exp()), dim=1)
    # each_data_KLD  = 0.5*torch.sum((- logvar + mu.pow(2) + logvar.exp()), dim=1)
    each_data_KLD  = 0.5*torch.sum((- torch.log(var+1e-16) + mu.pow(2) + var), dim=1)
    KLD = torch.sum(each_data_KLD)

    reg_term_decoder = 0.5*reg_param*torch.sum(model.dec.weight.flatten()**2)
    reg_term_encoder = 0.5*reg_param*torch.sum(model.mu.weight.flatten()**2)
    return beta*KLD + recon_loss + reg_term_decoder + reg_term_encoder, recon_loss, KLD

def criterion(model, hat_x, x, mu, var, beta=1.0, reg_param=1.0):

    first_term = torch.sum(x**2, dim=1)
    second_term = -2*torch.sum(hat_x*x, dim=1)
    third_term = torch.diag(mu @ model.dec.weight.T @ model.dec.weight @ mu.T)/x.size(1)
    forth_term = torch.sum(torch.diag(model.dec.weight.T @ model.dec.weight)*var)/x.size(1)
    each_data_recon_loss = 0.5*(first_term+second_term+third_term+forth_term)
    recon_loss = torch.sum(each_data_recon_loss)

    each_data_KLD  = 0.5*torch.sum((- torch.log(var+1e-16) + mu.pow(2) + var), dim=1)
    KLD = torch.sum(each_data_KLD)
    reg_term_decoder = 0.5*reg_param*torch.sum(torch.diag(model.dec.weight.T @ model.dec.weight))
    reg_term_encoder = 0.5*reg_param*torch.sum(torch.diag(model.mu.weight @ model.mu.weight.T))

    return beta*KLD+recon_loss+reg_term_decoder+reg_term_encoder, recon_loss, KLD


def batchfit_linearvae(dataset, model, num_epoch=100, lr=0.001, beta=1.0, reg_param=1.0, check_interval=1000, device="cpu"):

    history = {
            "elbo":[],
            "kl":[],
            "rate":[],
            "m": [],
            "d": [],
            "Q": [],
            "E":[],
            "R": [],
            "D":[],
            "eg_order" :[]}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epoch):

        dataset = dataset.to(device).view(-1, dataset.size(1)).to(torch.float32)
        recon, mu, var = model(dataset)
        loss, recon, kl = criterion(model, recon, dataset, mu, var, beta=beta, reg_param=reg_param)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ob_state = utils.calc_observable(model, 
                                        W0=None, 
                                        device=device)

        history["elbo"].append(loss.item())
        history["kl"].append(kl.item())
        history["rate"].append(recon.item())
        history["m"].append(ob_state[0])
        history["d"].append(ob_state[1])
        history["Q"].append(ob_state[2])
        history["E"].append(ob_state[3])
        history["R"].append(ob_state[4])
        history["D"].append(ob_state[5])
        history["eg_order"].append(ob_state[6])

        if epoch%check_interval==0:
            print(f'【EPOCH {epoch}】 eg: {ob_state[6]:.4f}, (elbo, recon, kl)=({loss.item():.4f}, {recon.item():0.4f}, {kl.item():0.4f})')

    return history, model