# Brain-Cognition modes of covariation

The instructions below allow to reproduce the work done in "A snapshot of brain and cognition in healthy mid-life and older adults" (2021) on the [PISA](https://doi.org/10.1016/j.nicl.2020.102527), [AIBL](https://aibl.csiro.au/) and [ADNI](http://adni.loni.usc.edu/) datasets.

## Step 1. Sulcal measurements
Follow the instructions [here](https://github.com/LeonieBorne/morpho-deepsulci-docker) to apply the [Morphologist pipeline](https://doi.org/10.1016/j.media.2020.101651) from the [Brainvisa toolbox](https://brainvisa.info) on your own dataset (T1w MRI scans).

## Step 2. Dataset organisation
Arrange your data so that you get the following three files: 
- ```<brain_csv>``` containing the brain measurements (e.g. sulcal width, cortical thickness, etc.).
- ```<cognition_csv>``` containing the cognitive scores.
- ```<info_csv>``` containing additional informations. It must contain "Age", "Sex" (1 if Male, 0 if Female) and "Group" (1 if healthy). It may contain "Amyloid" ("Positive" or "Negative"), "APOEe4" (4 if ε4-carrier else 0), "PRS_noAPOE" (polygenic risk score without APOE), "Twin" ("MZ" if monozygote) and "Twin_ID" (twin subject ID).

Each row correspond to a different participant, each column to a different feature.

## Step 3. Partial Least Square (PLS)

The scripts described below were developed and tested in Python 3.9.5. To run the scripts, the dependencies described in ```requirements.txt``` are required. The following command allow to install them: ``` pip install -r requirements.txt ```. [Docker](https://www.docker.com/) needs to be installed to create the sulci snapshots.

For more information on the scripts below and to see the options available, use the command ```python <script>.py -h```.

### Are the modes robust?

In order to see if the brain-cognition modes are robust, permutation tests are performed with the ```permutation.py``` script. 

The following command tests the PLS trained on the healthy cohort only:
```
python permutation.py <brain_csv> <cognition_csv> <info_csv> -m -t
```

The following command tests the PLS after regressing out age and sex:
```
python permutation.py <brain_csv> <cognition_csv> <info_csv> -m -r
```

### What are the contribution of each individual score (a specific cognitive test or brain measurement) to the shared variance?

This is checked using bootrapping tests on the [PLS loadings](https://scikit-learn.org/stable/modules/cross_decomposition.html#plscanonical), with the ```bootstrapping.py``` script:
```
python permutation.py <brain_csv> <cognition_csv> <info_csv> -m -t -f <figure_folder> --sulci_snapshot
```

### Plot age and group effects

Use the ```projections.py``` script to plot the age and group effects (sex, clinical status, amyloid status, APOE) as follows:

```
python projections.py <brain_csv> <cognition_csv> <info_csv> -m -t --plot_age_effect
```

### Sulci loadings comparison

Use the ```sulci_loadings_comparison.py``` script to create the snapshots comparing the sulci loadings from 2 different datasets (e.g. AIBL and ADNI), after bootstrapping:

```
python sulci_loadings_comparison.py <brain1_csv> <cognition1_csv> <info1_csv> <brain2_csv> <cognition2_csv> <info2_csv> -m --train_hc_only1 --train_hc_only2
```

### Is the specific composition of these modes implicitly optimized to covary with age?

To test whether any linear weighting of cognition and brain features would perform comparably well to covary with age, permutation on the features are used with the ```features_permutation.py``` script:
```
python features_permutation.py <brain_csv> <cognition_csv> <info_csv> -m -t -f <figure_file>
```

### Are the features identified by performing PLS on the healthy participants optimized to identify out-of-sample clinical participants (MCI/AD)?

The previous command also allows you to test if the linear weighting is optimized to identify out-of-sample participants.

### Which projection performs a better differentiation healthy from clinical participants?

The script ```classification.py``` tests how well a linear SVM trained on the projections allow to classify healthy from clinical participants.
The ```--brain2``` argument allows to compare the use of two different brain latent variables.
```
python classification.py <brain_csv> <cognition_csv> <info_csv> -m -t --brain2 <brain2_csv>
```
