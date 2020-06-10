# ExGAN

This repository contains the code for ExGAN: Adversarial Generation of Extreme Samples

## Getting Started

### Reproducing the Experiments

The first step is to get the data. We have prepared a script to download precipitation data from water.weather.gov/precip/. The data is
for the duration mentioned in the paper.

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

### Evaluation and Visualizing the Results

We provide FID.py to calculate the FID score, as described in the paper, on the trained models. 

The python file, Plot.py, contains the code for plotting rainfall maps like the ones included in the paper. 

We also provide an IPython notebook, EVT_Analysis.ipynb to play with and visualize the effect of different thresholds for the Peaks over 
Threshold approach.
