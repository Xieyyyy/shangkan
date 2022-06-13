import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchdiffeq


class Model(nn.Module):
    def __init__(self, datasets, args):
        super(Model, self).__init__()
        self.args = args
        self.linear = nn.Linear(1, self.args.HIDDEN_DIM)
        self.encoder = Encoder(args)
        self.variation = Variation(args)

    def forward(self, x, adj_mx):
        x = self.linear(x)
        h = self.encoder(x, adj_mx)
        latent_var, z_mu, z_sigma = self.variation(h)

#-----------------
class ODEDecoderDynamic(nn.Module):
    def __init__(self, ode_func, rtol=.01, atol=.001, method='dopri5', adjoint=False, terminal=False):
        super(ODEDecoderDynamic, self).__init__()
        self.ode_func = ode_func
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal
        self.perform_num = 0

    def forward(self, vt, y0):
        self.perform_num += 1
        integration_time_vector = vt.type_as(y0)
        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(func=self.ode_func, y0=y0, t=integration_time_vector, rtol=self.rtol,
                                             atol=self.atol, method=self.method)
        else:
            out = torchdiffeq.odeint(func=self.ode_func, y0=y0, t=integration_time_vector, rtol=self.rtol,
                                     atol=self.atol, method=self.method)
        return out

    def reset(self):
        self.perform_num = 0


# -----------------
class Variation(nn.Module):
    def __init__(self, args):
        super(Variation, self).__init__()
        self.enc_mu = nn.Linear(args.HIDDEN_DIM, args.HIDDEN_DIM)
        self.enc_sigma = nn.Linear(args.HIDDEN_DIM, args.HIDDEN_DIM)

    def forward(self, h):
        mu = self.enc_mu(h)
        log_sigma = self.enc_sigma(h)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(mu.device)
        return mu + sigma * nn.Parameter(std_z, requires_grad=False), mu, sigma


# -------------------
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.encoder_unit = Encoder_Unit(args)

    def forward(self, x, adj_mx):
        L = x.shape[1]
        encoder_hidden_state = None
        for t in range(L):
            output, encoder_hidden_state = self.encoder_unit(x[:, t, ...], adj_mx, encoder_hidden_state)
        return output


class Encoder_Unit(nn.Module):
    def __init__(self, args):
        super(Encoder_Unit, self).__init__()
        self.args = args
        self.gru_layers = nn.ModuleList([GRU_Layer(args) for _ in range(self.args.RNN_LAYER)])

    def forward(self, x, adj_mx, hidden_state=None):
        B = x.shape[0]
        if hidden_state == None:
            hidden_state = torch.zeros([self.args.RNN_LAYER, x.shape[0], x.shape[1], x.shape[2]],
                                       device=self.args.DEVICE)
        hidden_states = []
        output = x
        for layer_num, dcgru_layer in enumerate(self.gru_layers):
            next_hidden_state = dcgru_layer(output, adj_mx, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class GRU_Layer(nn.Module):
    def __init__(self, args):
        super(GRU_Layer, self).__init__()
        self.args = args
        self.z_gcn = GCN(args, self.args.HIDDEN_DIM * 2, self.args.HIDDEN_DIM)
        self.r_gcn = GCN(args, self.args.HIDDEN_DIM * 2, self.args.HIDDEN_DIM)
        self.z_gcn = GCN(args, self.args.HIDDEN_DIM * 2, self.args.HIDDEN_DIM)

    def forward(self, x, adj_mx, hidden_states):
        input_and_state = torch.cat([x, hidden_states], dim=-1)
        z = self.z_gcn(input_and_state, adj_mx, activation=F.sigmoid)
        r = self.r_gcn(input_and_state, adj_mx, activation=F.sigmoid)
        h = self.z_gcn(torch.cat([x, hidden_states * r], dim=-1), adj_mx, activation=F.tanh)
        h = (1 - z) * hidden_states + z * h
        return h


class GCN(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GCN, self).__init__()
        self.args = args
        self.transformation = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj_mx, activation):
        adj_mul_x = torch.matmul(adj_mx, x)
        value = activation(self.transformation(adj_mul_x))

        return value
