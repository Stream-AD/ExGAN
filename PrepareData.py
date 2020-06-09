from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from urllib.request import urlretrieve
import os
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import LongTensor, FloatTensor

curdt = datetime(2010,1,1) #Start data collection from this date

os.makedirs('data', exist_ok=True)
### Alternate URL for downloading data (not currently in use)
# url = 'http://water.weather.gov/precip/downloads/{dt:%Y/%m/%d}/nws_precip_1day_'\
      # '{dt:%Y%m%d}_conus.nc'.format(dt=dt)

### Download data
for i in range(2557):
    try:
        url = 'http://water.weather.gov/precip/archive/{dt:%Y/%m/%d}/nws_precip_conus_{dt:%Y%m%d}.nc'.format(dt=curdt)
        urlretrieve(url, 'data/nws_precip_conus_{dt:%Y%m%d}.nc'.format(dt=curdt))
        curdt = curdt + timedelta(days=i)
    except:
        pass



class NWSDataset(Dataset):
    """
    NWS Dataset
    """

    def __init__(
        self, path='data', prefix="nws_precip_conus_"
    ):
        self.path = path
        self.files = [
                        f
                        for f in os.listdir(path)
                        if f.startswith(prefix) and os.path.isfile(os.path.join(path, f))
                    ]
        self.maxclip = 100

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        nc = NetCDFFile(
            os.path.join(self.path,self.files[item])
        )
        prcpvar = nc.variables["amountofprecip"]
        data = 0.01 * (prcpvar[:] + 1)
        data = FloatTensor(resize(data, (64, 64)))
        data = data.view((-1, 64, 64))
        data = torch.clamp(data, max=self.maxclip)
        data = data / self.maxclip
        data = (data * 2) - 1  # Between -1 and 1
        return data

dataloader = DataLoader(NWSDataset(), batch_size=256, shuffle=True)
data = []
for i in dataloader:
    data.append(i)
data = torch.cat(data, 0)
sums = data.sum(dim = (1, 2, 3)).detach().cpu().numpy().argsort()[::-1].copy()
torch.save(data.data[sums], 'data/real.pt')