import torch

def generate_data_form_SCM(N, M, W0=None, eta=0.5, rho=1.0, device="cpu"):
    if W0==None:
        W0 = torch.ones((N, M))
    # generate random noise
    c = torch.randn(M, 1, device=device)
    n = torch.randn(N, 1, device=device)

    X = (W0@c)/(torch.sqrt(torch.tensor(N/rho))) + torch.sqrt(torch.tensor(eta))*n
    return X.T

def make_dataset_from_SCM(P, N, M, W0=None, eta=0.5, rho=1.0, device="cpu"):
    dataset=torch.vstack([generate_data_form_SCM(N=N, M=M, W0=W0, eta=eta, rho=rho, device=device) for _ in range(P)])
    return dataset