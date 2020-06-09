from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import LongTensor, FloatTensor
from torchvision.utils import save_image
import sys

class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(
        self, fake='data/fake.pt', c=0.75, i=1, n=2557
    ):
        val = int(n*(c**i))
        self.real = torch.load('data/real.pt').cuda()
        self.fake = torch.load(fake).cuda()
        self.realdata = torch.cat([self.real[:val], self.fake[:-1*val]], 0)
        self.realdata.requires_grad = False
        
    def __len__(self):
        return self.realdata.shape[0]

    def __getitem__(self, item):
        return self.realdata[item]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def convTBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


def convBNReLU(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, True),
    )


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = convTBNReLU(in_channels, 512, 4, 1, 0)
        self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    def forward(self, noise):
        out = self.block1(noise)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return torch.tanh(self.block5(out))


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.block1 = convBNReLU(self.in_channels, 64)
        self.block2 = convBNReLU(64, 128)
        self.block3 = convBNReLU(128, 256)
        self.block4 = convBNReLU(256, 512)
        self.block5 = nn.Conv2d(512, 64, 4, 1, 0)
        self.source = nn.Linear(64, 1)

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        out = out.view(size, -1)
        source = torch.sigmoid(self.source(out))
        return source
    
latentdim = 20
criterionSource = nn.BCELoss()
G = Generator(in_channels=latentdim, out_channels=1).cuda()
D = Discriminator(in_channels=1).cuda()
G.apply(weights_init_normal)
D.apply(weights_init_normal)

optimizerG = optim.Adam(G.parameters(), lr=0.00002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))
static_z = Variable(FloatTensor(torch.randn((81, latentdim, 1, 1)))).cuda()

def sample_image(stage, epoch):
    static_sample = G(static_z).detach().cpu()
    static_sample = (static_sample + 1) / 2.0
    save_image(static_sample, DIRNAME + "stage%depoch%d.png" % (stage, epoch), nrow=9)

DIRNAME = sys.argv[1] + '/'
os.makedirs(DIRNAME, exist_ok=True)
board = SummaryWriter(log_dir=DIRNAME)

G.load_state_dict(torch.load('Generator.pt'))
D.load_state_dict(torch.load('Discriminator.pt'))
step = 0
fake_name = 'data/fake.pt'
c = 0.75
k = 10
n = 2557
for i in range(1, k+1):
    dataloader = DataLoader(NWSDataset(fake=fake_name, c=c, i=i, n=n), batch_size=256, shuffle=True)
    for epoch in range(0, 100):
        print(epoch)
        for realdata in dataloader:
            noise = 1e-5*max(1 - (epoch/100.0), 0)
            step += 1
            batch_size = realdata[0].shape[0]
            trueTensor = 0.7 + 0.5 * torch.rand(batch_size)
            falseTensor = 0.3 * torch.rand(batch_size)
            probFlip = torch.rand(batch_size) < 0.05
            probFlip = probFlip.float()
            trueTensor, falseTensor = (
                probFlip * falseTensor + (1 - probFlip) * trueTensor,
                probFlip * trueTensor + (1 - probFlip) * falseTensor,
            )
            trueTensor = trueTensor.view(-1, 1).cuda()
            falseTensor = falseTensor.view(-1, 1).cuda()
            realdata = realdata.cuda()
            realSource = D(realdata)
            realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
            latent = Variable(torch.randn(batch_size, latentdim, 1, 1)).cuda()
            fakeGen = G(latent)
            fakeGenSource = D(fakeGen.detach())
            fakeGenLoss = criterionSource(fakeGenSource, falseTensor.expand_as(fakeGenSource))
            lossD = realLoss + fakeGenLoss
            optimizerD.zero_grad()
            lossD.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(),20)
            optimizerD.step()
            fakeGenSource = D(fakeGen)
            lossG = criterionSource(fakeGenSource, trueTensor.expand_as(fakeGenSource))
            optimizerG.zero_grad()
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(),20)
            optimizerG.step() 
            board.add_scalar('realLoss', realLoss.item(), step)
            board.add_scalar('fakeGenLoss', fakeGenLoss.item(), step)
            board.add_scalar('lossD', lossD.item(), step)
            board.add_scalar('lossG', lossG.item(), step)
        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), DIRNAME + "Gstage"+str(i) + 'epoch' + str(epoch) + ".pt")
            torch.save(D.state_dict(), DIRNAME + "Dstage"+str(i) + 'epoch' + str(epoch) + ".pt")
        if (epoch + 1) % 10 == 0:   
            with torch.no_grad():
                G.eval()
                sample_image(i, epoch)
                G.train()
    with torch.no_grad():
        G.eval()
        fsize = int((1-(c**(i+1)))*n/c)
        print(fsize)
        fakeSamples = G(Variable(torch.randn(fsize, latentdim, 1, 1)).cuda())
        sums = fakeSamples.sum(dim = (1, 2, 3)).detach().cpu().numpy().argsort()[::-1].copy()
        fake_name = DIRNAME+'fake'+str(i+1)+'.pt'
        torch.save(fakeSamples.data[sums], fake_name)
        del fakeSamples
        G.train()