import torch
import torch.nn as nn
# from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        # self.main = nn.Sequential(
        #     nn.Linear(4, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 2)
        # )
        self.main = nn.Sequential(
            nn.Linear(1, 1)
        )

    # def _dist(self, s):
    #     s = torch.FloatTensor(s)
    #     return Categorical(logits=self.main(s))

    # def forward(self, s):
    #     return self._dist(s).sample()

    def forward(self, s):
        return self.main(torch.FloatTensor(s))