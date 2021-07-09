# Brain-Cognition modes of covariation 

The instructions below allow to reproduce the work done in XXX on the [PISA](https://doi.org/10.1016/j.nicl.2020.102527), [AIBL](https://aibl.csiro.au/) and [ADNI](http://adni.loni.usc.edu/) datasets.

The file [figures_article.py](figures_article.py) allow to reproduce the figures in XXX.

## Step 1. Brain measurements
Follow the instructions [here](https://github.com/LeonieBorne/morpho-deepsulci-docker) to apply the [Morphologist pipeline](https://doi.org/10.1016/j.media.2020.101651) from the [Brainvisa toolbox](https://brainvisa.info) on your own dataset (T1w MRI scans).

## Step 2. Dataset organisation
Arrange your data so that you get the following three files: 
- ```<sulci_csv>```
- ```<cognition_csv>```
- ```<info_csv>```

## Step 3. Partial Least Square (PLS)
You are ready to run the scripts in [figures_article.py](figures_article.py)!
