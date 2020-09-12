# ExGAN

This repository contains the code for ExGAN: Adversarial Generation of Extreme Samples

## Getting Started

### Environment
This code has been tested on Debian GNU/Linux 9 with a 12GB Nvidia GeForce RTX 2080 Ti GPU, CUDA Version 10.2 and PyTorch 1.5.  

### Reproducing the Experiments

The first step is to get the data. We have prepared a script to download precipitation data from [water.weather.gov/precip/](https://water.weather.gov/precip/). The data downloaded is for the duration 2010 to 2016 as mentioned in the paper.

```
python PrepareData.py
```
Now, we can train a DCGAN Baseline on this data. 

```
python DCGAN.py
```
Distribution Shifting on this DCGAN can be performed using 
```
python DistributionShifting.py
```
Finally, we can train ExGAN on the distribution shifted dataset. 
```
python ExGAN.py
```

The training of ExGAN and DCGAN can be monitored using TensorBoard. 
```
tensorboard --logdir [DCGAN\EXGAN]
```

### Evaluation and Visualizing the Results

Generate samples from DCGAN of different extremeness probabilities, and mark the time taken in sampling.
```
python DCGANSampling.py
```
Similarly, Generate samples from ExGAN of different extremeness probabilities, and mark the time taken in sampling.
```
python ExGANSampling.py
```

We provide FID.py to calculate the FID score, as described in the paper, on the trained models. 
We also provide DCGANRecLoss.py, and ExGANRecLoss.py to evaluate DCGAN and ExGAN on their Reconstruction Loss
Note that, both of these metrics are calculated on a test set. PrepareData.py can be used to curate the test set for the duration described in the paper.

The python file, plot.py, contains the code for plotting rainfall maps like the figures included in the paper. Note that this requires the Basemap library from matplotlib. 

We also provide an IPython notebook, EVT_Analysis.ipynb to play with and visualize the effect of different thresholds for the Peaks over 
Threshold approach.
