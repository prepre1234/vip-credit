import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from datetime import datetime
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32, help='size of the batches')
parser.add_argument('--epo', type=int, default=1, help='size of epoches')
parser.add_argument('--dim', type=int, default=10, help='dim of a single data')
opt = parser.parse_args()
print('args:')
print(opt)


class dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        data = self.data[index % self.data.shape[0]]
        return data

    def __len__(self):
        return self.data.shape[0]


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *block(dim, 32),
            *block(32, 64),
            *block(64, 128),
            *block(128, 64),
            *block(64, 32),
            nn.Linear(32, dim),
            # Tanh()要加吗？
            nn.Tanh()
        )

    def forward(self, x):
        gen_x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.Global = nn.Sequential(
            nn.Linear(dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.Local = nn.Sequential(
            nn.Linear(5, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.combine = nn.Sequential(
            nn.Linear(96, 32),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        cat = torch.cat((self.Global(x), self.Local(
            x[:, 2:7]), self.Local(x[:, 5:])), axis=1)
        return self.combine(cat)


# Loss function
criterion_gan = torch.nn.MSELoss()
criterion_identity = torch.nn.L1Loss()

# Initialize generator and discrimimator
generator = Generator(opt.dim)
discriminator = Discriminator(opt.dim)

# Optimizer
optimizer_g = torch.optim.Adam(
    generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(
    discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion_gan.cuda()
    criterion_identity()

data = pd.read_csv('./dataDump/scaler_data.csv').iloc[:, 1:]
dataNotNull = data.dropna()
dataNull = data[data.isnull().T.any()]

dataloader = DataLoader(
    # remove labels
    dataset(np.array(dataNotNull.iloc[:, 1:])),
    batch_size=opt.bs,
    shuffle=True,
    drop_last=True
)


print('training...')
begin = datetime.now()

# Begin training
for epoch in range(opt.epo):
    for i, batch in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(opt.bs, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(opt.bs, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real = Variable(batch.type(Tensor))
        deficiency = Variable(batch.type(Tensor))
        deficiency[:, [4, 9]] = 0

        # ----------------
        # Train generator
        # ----------------

        optimizer_g.zero_grad()

        # Complete the deficient data
        completion = generator(deficiency)

        # Gan loss. Loss measures generator's ability to fool the discriminator
        loss_gan = criterion_gan(discriminator(completion), valid)

        # Identity loss. Cosistence
        loss_identity = criterion_identity(completion, real)

        # Total loss
        loss_g = loss_gan+5*loss_identity

        loss_g.backward()
        optimizer_g.step()

        # --------------------
        # Train Discriminator
        # --------------------

        optimizer_d.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = criterion_gan(discriminator(real), valid)
        fake_loss = criterion_gan(discriminator(completion.detach()), fake)
        loss_d = (real_loss+fake_loss)/2

        loss_d.backward()
        optimizer_d.step()

        # print log
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.epo, i, len(dataloader), loss_d.item(), loss_g.item())
        )

    end = datetime.now()
    print("Total time", (end-begin).seconds)


# ------------ Completing -------------
print('completing...')
begin = datetime.now()

generator.eval()

dataNullLabel = dataNull.iloc[:, 0]
dataNullInfo = dataNull.iloc[:, 1:]
dataNullInfo = np.array(dataNullInfo)

for i in range(dataNullInfo.shape[0]):
    print(i)
    if np.isnan(dataNullInfo[i, 4]) and np.isnan(dataNullInfo[i, 9]):
        dataNullInfo[i, [4, 9]] = 0
        gen = generator(
            Variable(Tensor(dataNullInfo[i]).view(1, opt.dim))).view(opt.dim)
        dataNullInfo[i, [4, 9]] = gen[[4, 9]]
    elif np.isnan(dataNullInfo[i, 4]):
        dataNullInfo[i, 4] = 0
        gen = generator(
            Variable(Tensor(dataNullInfo[i]).view(1, opt.dim))).view(opt.dim)
        dataNullInfo[i, 4] = gen[4]
    else:
        dataNullInfo[i, 9] = 0
        gen = generator(
            Variable(Tensor(dataNullInfo[i]).view(1, opt.dim))).view(opt.dim)
        dataNullInfo[i, 9] = gen[9]

end = datetime.now()
print("Total time", (end-begin).seconds)

dataNullCompletion = pd.concat(
    [pd.DataFrame(np.array(dataNullLabel)), pd.DataFrame(dataNullInfo)], axis=1, ignore_index=True)

dataCompletion = pd.concat([pd.DataFrame(np.array(dataNotNull)), pd.DataFrame(
    np.array(dataNullCompletion))], axis=0)

dataCompletion.to_csv('./dataDump/completion_data.csv')
