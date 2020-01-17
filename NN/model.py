from torch import nn


class network(nn.Module):
    def __init__(self,dim):
        super(network, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0, 8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(dim, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 64),
            *block(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
