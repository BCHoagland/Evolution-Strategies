from collections import OrderedDict
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Model(nn.Module):
    def __init__(self, n_in, n_h, n_out):
        super().__init__()

        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out

    def num_params(self):
        l1 = (self.n_in * self.n_h) + self.n_h 
        l2 = (self.n_h * self.n_h) + self.n_h
        l3 = (self.n_h * self.n_out) + self.n_out
        return l1 + l2 + l3

    def params2layers(self, params):
        l1 = self.n_h * self.n_in
        l2 = l1 + self.n_h
        l3 = l2 + (self.n_h * self.n_h)
        l4 = l3 + self.n_h
        l5 = l4 + (self.n_h * self.n_out)
        return OrderedDict({
            'main.0.weight': torch.tensor(params[:l1]).view(self.n_h, self.n_in),
            'main.0.bias': torch.tensor(params[l1:l2]),
            'main.2.weight': torch.tensor(params[l2:l3]).view(self.n_h, self.n_h),
            'main.2.bias': torch.tensor(params[l3:l4]),
            'main.4.weight': torch.tensor(params[l4:l5]).view(self.n_out, self.n_h),
            'main.4.bias': torch.tensor(params[l5:]),
        })



class MLP(Model):
    def __init__(self, n_in, n_h, n_out):
        super().__init__(n_in, n_h, n_out)

        self.main = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_out)
        )

    def set_params(self, params):
        p = self.params2layers(params)
        self.load_state_dict(p, strict=True)

    def forward(self, s):
        logits = self.main(torch.FloatTensor(s))
        dist = Categorical(logits=logits)
        return dist.sample()
