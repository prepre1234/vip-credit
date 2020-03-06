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
parser.add_argument('--dim', type=int, default=10, help='dim of data')
parser.add_argument('--n_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--latent_dim', type=int, default=32,
                    help='dim of latent space')
parser.add_argument('--embDim', type=int, default=2,
                    help='dim of embedding vector')
opt = parser.parse_args()
print('args:')
print(opt)


class dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        if self.data.shape[0] != self.labels.shape[0]:
            raise Exception(
                'number of the data should be equal to number of labels!')

    def __getitem__(self, index):
        data = self.data[index % self.data.shape[0]]
        label = self.labels[index % self.data.shape[0]]

        return {"data": data, "labels": label}

    def __len__(self):
        return self.data.shape[0]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.embDim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim+opt.embDim, 64, normalize=False),
            *block(64, 128),
            *block(128, 128),
            *block(128, 64),
            nn.Linear(64, opt.dim),
            # Tanh?????
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate noise vector and label embedding -> input
        input = torch.cat((self.label_emb(labels), noise), 1)

        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.embDim)

        self.model = nn.Sequential(
            nn.Linear(opt.embDim+opt.dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, data, labels):
        input = torch.cat((data, self.label_emb(labels)), 1)

        return self.model(input)


# Loss function
criterion = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizer
optimizer_g = torch.optim.Adam(
    generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(
    discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()

data = pd.read_csv('./dataDump/completion_data.csv').iloc[:, 1:]
labels = np.array(data.iloc[:, 0]).reshape(data.shape[0], -1)
data_train = np.array(data.iloc[:, 1:])

dataloader = DataLoader(
    dataset(data_train, labels),
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
        valid = Variable(FloatTensor(opt.bs, 1).fill_(1.0),
                         requires_grad=False)
        fake = Variable(FloatTensor(opt.bs, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real = Variable(batch['data'].type(FloatTensor))
        label = Variable(batch['labels'].flatten().type(LongTensor))

        # ----------------
        # Train generator
        # ----------------

        optimizer_g.zero_grad()

        # Sample noise and labels as generator's input
        z = Variable(FloatTensor(np.random.normal(
            0, 1, (opt.bs, opt.latent_dim))))
        gen_labels = Variable(LongTensor(
            np.random.randint(0, opt.n_classes, opt.bs)))

        gen_data = generator(z, gen_labels)

        # Gan loss. Loss measures generator's ability to fool the discriminator
        loss_g = criterion(discriminator(gen_data, gen_labels), valid)

        loss_g.backward()
        optimizer_g.step()

        # --------------------
        # Train Discriminator
        # --------------------

        optimizer_d.zero_grad()

        real_loss = criterion(discriminator(real, label), valid)
        fake_loss = criterion(discriminator(
            gen_data.detach(), gen_labels), fake)

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


# -----------------balance----------------
print('balance')

generator.eval()

z = Variable(FloatTensor(np.random.normal(0, 1, (20000, opt.latent_dim))))
gen_labels = Variable(LongTensor(np.ones(20000)))
gen = generator(z, gen_labels)
gen = np.concatenate(
    (np.array(gen_labels).reshape(20000, 1), gen.detach().numpy()), axis=1)

balance = np.concatenate((np.array(data), gen))
pd.DataFrame(balance).to_csv('./dataDump/balance_data.csv')
