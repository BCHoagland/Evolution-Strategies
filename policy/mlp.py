from collections import OrderedDict
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # print([(k, v.shape) for (k, v) in self.state_dict().items()])
        # quit()

        d = OrderedDict({
            'main.0.weight': torch.tensor(params[:32]).view(32, 1),
            'main.0.bias': torch.tensor(params[32:64]),
            'main.2.weight': torch.tensor(params[64:1088]).view(32, 32),
            'main.2.bias': torch.tensor(params[1088:1120]),
            'main.4.weight': torch.tensor(params[1120:1152]).view(1, 32),
            'main.4.bias': torch.tensor(params[1152]).unsqueeze(0)
        })
        self.load_state_dict(d, strict=True)

    def forward(self, s):
        with torch.no_grad():
            return self.main(torch.FloatTensor([s])).numpy()
