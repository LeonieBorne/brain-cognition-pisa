# Brain-Cognition modes of covariation (WORK IN PROGRESS)

The instructions below allow to reproduce the work done in XXX on the [PISA](https://doi.org/10.1016/j.nicl.2020.102527), [AIBL](https://aibl.csiro.au/) and [ADNI](http://adni.loni.usc.edu/) datasets.

## Step 1. Sulcal measurements
Follow the instructions [here](https://github.com/LeonieBorne/morpho-deepsulci-docker) to apply the [Morphologist pipeline](https://doi.org/10.1016/j.media.2020.101651) from the [Brainvisa toolbox](https://brainvisa.info) on your own dataset (T1w MRI scans).

## Step 2. Dataset organisation
Arrange your data so that you get the following three files: 
- ```<brain_csv>```
- ```<cognition_csv>```
- ```<info_csv>```
Each row correspond to a different participant, each column a different feature.

## Step 3. Partial Least Square (PLS)

Install the following library (...) 

### Are the modes robust?

### What are the contribution of each individual score (a specific cognitive test or brain measurement) to the shared variance?

### Train the model and apply on (new) data

### Plot age and group effects

### Is the specific composition of these modes implicitly optimized to covary with age?
To test whether any linear weighting of poor cognition and wider sulci would perform comparably well

### Are the features identified by performing PLS on the healthy participants optimized to identify out-of-sample clinical participants (MCI/AD)?

### Which projection performs a better differentiation healthy from clinical participants?
