from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()
 
    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x