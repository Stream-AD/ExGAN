from tensorboardX import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from skimage.transform import resize
from torchvision import transforms
import torch.optim as optim
from torch import LongTensor, FloatTensor
from scipy.stats import skewnorm, genpareto
from torchvision.utils import save_image
import sys

class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(
        self
    ):
        self.real = torch.load('data/real.pt').cuda()
        self.lbls = 20*(self.real.sum(dim=(1, 2, 3))/4096)+19
        
    def __len__(self):
        return self.real.shape[0]

    def __getitem__(self, item):
        return self.real[item], self.lbls[item].view(-1, 1)


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
        self.block1 = convTBNReLU(in_channels + 1, 512, 4, 1, 0)
        self.block2 = convTBNReLU(512, 256)
        self.block3 = convTBNReLU(256, 128)
        self.block4 = convTBNReLU(128, 64)
        self.block5 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

    def forward(self, latent, continuous_code):
        inp = torch.cat((latent, continuous_code), 1)
        out = self.block1(inp)
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
        self.source = nn.Linear(64+1, 1)

    def forward(self, inp, extreme):
        out = self.block1(inp)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        size = out.shape[0]
        out = out.view(size, -1)
        sums = 20*(inp.sum(dim=(1, 2, 3))/4096)+19
        diff = extreme.view(size, 1) - sums.view(size, 1)
        source = torch.sigmoid(self.source(torch.cat([out, diff.view(size, 1)], 1)))
        return source

latentdim = 20
criterionSource = nn.BCELoss()
criterionContinuous = nn.L1Loss()
criterionValG = nn.L1Loss()
criterionValD = nn.L1Loss()
G = Generator(in_channels=latentdim, out_channels=1).cuda()
D = Discriminator(in_channels=1).cuda()
G.apply(weights_init_normal)
D.apply(weights_init_normal)
skewnorm_params = (10.732322904924764, -0.9232793649630112, 0.5328454693681157)
rv = skewnorm(*skewnorm_params)

def sample_skewnorm(size):
    return FloatTensor(rv.ppf(FloatTensor(*size).uniform_(0, 1)))

def sample_cont_code(batch_size):
    return Variable(sample_skewnorm((batch_size, 1, 1, 1))).cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
static_code = sample_cont_code(81)

def sample_image(batches_done):
    static_z = Variable(FloatTensor(torch.randn((81, latentdim, 1, 1)))).cuda()
    static_sample = G(static_z, static_code).detach().cpu()
    static_sample = (static_sample + 1) / 2.0
    save_image(static_sample, DIRNAME + "%d.png" % batches_done, nrow=9)

    c_varied = np.linspace(0.01, 0.99, 81)[np.newaxis, :, np.newaxis, np.newaxis].reshape(81, 1, 1, 1)
    c1 = Variable(FloatTensor(rv.ppf(c_varied))).cuda()
    sample1 = G(static_z, c1).detach().cpu()
    new = sample1.sum(dim=(1, 2, 3)) / 4096.0
    new = new*20 + 19
    new = new.view(-1).cpu()
    old = c1.view(-1).cpu()
    diffs = torch.abs((new - old) / (old + 1e-16))
    print(diffs.mean(), diffs.max(), diffs.min(), diffs.std())
    save_image(0.5*(sample1+1), DIRNAME + "c1-%d.png" % batches_done, nrow=9)

DIRNAME = sys.argv[1] + '/'
os.makedirs(DIRNAME, exist_ok=True)
board = SummaryWriter(log_dir=DIRNAME)
step = 0
dataloader = DataLoader(NWSDataset(), batch_size=256, shuffle=True)
for epoch in range(0, 1000):
    print(epoch)
    for images, labels in dataloader:
        noise = 1e-5*max(1 - (epoch/1000.0), 0)
        step += 1
        batch_size = images.shape[0]
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
        images, labels = images.cuda(), labels.cuda()
        realSource = D(images, labels)
        realLoss = criterionSource(realSource, trueTensor.expand_as(realSource))
        latent = Variable(torch.randn(batch_size, latentdim, 1, 1)).cuda()
        code = sample_cont_code(batch_size)
        fakeGen = G(latent, code)
        fakeGenSource = D(fakeGen.detach(), code.detach())
        fakeGenLoss = criterionSource(fakeGenSource, falseTensor.expand_as(fakeGenSource))
        lossD = realLoss + fakeGenLoss
        optimizerD.zero_grad()
        lossD.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(),20)
        optimizerD.step()
        fakeGenSource = D(fakeGen, code)
        fakeLabels = 20*(fakeGen.sum(dim=(1, 2, 3))/4096) + 19
        rpd = torch.mean(torch.abs((fakeLabels - code.view(batch_size)) / (code.view(batch_size) + 1e-16)))
        lossG = criterionSource(fakeGenSource, trueTensor.expand_as(fakeGenSource)) + rpd
        optimizerG.zero_grad()
        lossG.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(),20)
        optimizerG.step()
        board.add_scalar('realLoss', realLoss.item(), step)
        board.add_scalar('fakeGenLoss', fakeGenLoss.item(), step)
        board.add_scalar('fakeContLoss', rpd.item(), step)
        board.add_scalar('lossD', lossD.item(), step)
        board.add_scalar('lossG', lossG.item(), step)
    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), DIRNAME + 'Gepoch' + str(epoch) + ".pt")
        torch.save(D.state_dict(), DIRNAME + 'Depoch' + str(epoch) + ".pt")
    if (epoch + 1) % 10 == 0:   
        with torch.no_grad():
            G.eval()
            sample_image(epoch)
            G.train()