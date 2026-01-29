import torch.nn as nn

class RegressionMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(384, 192, 96), dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),          
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        
        layers.append(nn.Linear(prev, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)