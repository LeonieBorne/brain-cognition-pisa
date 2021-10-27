#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import functions.preprocessing
import functions.pls
import functions.snapshots
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.impute import SimpleImputer
import os
import numpy as np
import seaborn as sns
from sklearn.utils import resample

# command line arguments
parser = argparse.ArgumentParser(
    description='Sulci loadings comparison figures.')

parser.add_argument("brain1", help="Path to csv file with 1rst brain data.")
parser.add_argument("cognition1", help="Path to csv file with 1rst cognition data.")
parser.add_argument("info1", help="Path to csv file with 1rst additional information, containing age ('Age' or age_col), sex ('Sex' or sex_col; 1 if Male, 0 if Female) and clinical status ('Group' or group_col; 1 if healthy).")
parser.add_argument("--train_hc_only1", help="Train only on healthy participants for 1rst dataset.", action="store_true")
parser.add_argument("--regress_out_confounds1", help="Regress out confounds for 1rst dataset.", action="store_true")

parser.add_argument("brain2", help="Path to csv file with 2nd brain data.")
parser.add_argument("cognition2", help="Path to csv file with 2nd cognition data.")
parser.add_argument("info2", help="Path to csv file with 2nd additional information, containing age ('Age' or age_col), sex ('Sex' or sex_col; 1 if Male, 0 if Female) and clinical status ('Group' or group_col; 1 if healthy).")
parser.add_argument("--train_hc_only2", help="Train only on healthy participants for 2nd dataset.", action="store_true")
parser.add_argument("--regress_out_confounds2", help="Regress out confounds for 2nd dataset.", action="store_true")

parser.add_argument("-a", "--age_col", help="Column name in info csv with participants age", default="Age")
parser.add_argument("-s", "--sex_col", help="Column name in info csv with participants sex (1 if Male, 0 if Female)", default="Sex")
parser.add_argument("-g", "--group_col", help="Column name in info csv with participants clinical status (1 if healthy)", default="Group")
parser.add_argument("-m", "--mean", help="Average brain measurement (columns 'xxx_left' and 'xxx_right').", action="store_true")
parser.add_argument("-d", "--drop_rate", help="Maximum percentage of missing values without removing the subject/feature.", type=float, default=0.5)

# model options
parser.add_argument("--cca", help="Use CCA instead of PLS model", action="store_true")
parser.add_argument("-n", "--nboot", help="Number of bootstrapping tests", type=int, default=1000)
parser.add_argument("--mode", help="Tested mode (1rst: 0; 2nd: 1; etc.)", type=int, default=0)
# output arguments
parser.add_argument("-f", "--figure_folder", help="Output figure folder", default="")

args = parser.parse_args()

# load data
def check_csv(csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        return df
    else:
        print(f'CSV file not found: {csv_file}')
        exit(1)

df_brain1 = check_csv(args.brain1)
df_brain2 = check_csv(args.brain2)
df_cogn1 = check_csv(args.cognition1)
df_cogn2 = check_csv(args.cognition2)
df_info1 = check_csv(args.info1)
df_info2 = check_csv(args.info2)
    
# check arguments
def check_col(df_info):
    for col in [args.group_col, args.age_col, args.sex_col]:
        if col not in df_info.columns:
            print(f'Column {col} not in {args.info}.')
            exit(1)

check_col(df_info1)
check_col(df_info2)

# pipeline
if args.cca:
    print('Using CCA...')
    pipeline = functions.pls.PLSPipeline(CCA(n_components=args.mode+1, max_iter=1000),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
else:
    print('Using Canonical PLS...')
    pipeline = functions.pls.PLSPipeline(PLSCanonical(n_components=args.mode+1, max_iter=1000),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))


# loadings comparison snapshots
remove_missing = True
downsampling = True
mean = True
cogn_startswith = None

boostx = []
for df_brain, df_cogn, df_info, regress_out_confounds, train_hc_only in zip(
        [df_brain1, df_brain2], [df_cogn1, df_cogn2], [df_info1, df_info2],
        [args.regress_out_confounds1, args.regress_out_confounds2],
        [args.train_hc_only1, args.train_hc_only2]):
    dfb, dfc, dfi, Xbrain, Ycogn = functions.preprocessing.preprocessing(
        df_brain, df_cogn, df_info, 
        group_col=args.group_col, age_col=args.age_col, sex_col=args.sex_col,
        drop_rate=args.drop_rate, regress_out_confounds=regress_out_confounds,
        train_hc_only=train_hc_only, mean=args.mean, verbose=False)
    np.random.seed(0)
    x_loadings = []
    for i in range(args.nboot):
        Xr, Yr = resample(Xbrain, Ycogn)
        Xrt, Yrt = pipeline.fit_transform(Xr, Yr)
        x_loadings.append(pipeline.PLS.x_loadings_)
    boostx.append(np.array(x_loadings))

# sulci snapshot
sc = np.array(boostx[0]).mean(axis=0)[:, args.mode] - np.array(boostx[1]).mean(axis=0)[:, args.mode]
dict_sulcus = {s+'_left': x for s,x in zip(Xbrain.columns, sc) if s!=0}
for s in Xbrain.columns:
    dict_sulcus[s+'_left'] = dict_sulcus[s+'_left']
dict_reg = {0 : [0.5, -0.5, -0.5, 0.5], 1 : [0.5, 0.5, 0.5, 0.5]}
for side in ['left']:
    for reg in [0, 1]:
        sfile = os.path.join(args.figure_folder, f'brain1-brain2_{side}{reg}.png')
        functions.snapshots.view_sulcus_scores(
            dict_sulcus, side=side, reg_q=dict_reg[reg],
            minVal=-0.07, maxVal=0.07, snapshot=sfile)
        print(f'Snapshot saved at {sfile}')

# correlations
corr = np.corrcoef([np.array(boostx[i]).mean(axis=0)[:, 0] for i in range(2)])
print(f'Loadings correlation: {corr[0,1]}')

# histogram
means = np.array(boostx).mean(axis=0).mean(axis=0)[:,0]
labels = Xbrain.columns
labels = [l for m, l in sorted(zip(means, labels))]

titles = ['Brain 1', 'Brain 2']
plt.rcParams.update({'font.size': 40})
fig, axes = plt.subplots(1, 2, figsize=(30, 40))
for i in range(2):
    ax = axes[i]
    x_loadings = np.array(boostx)[i]
    means_db = [m if up*down > 0 else 0 for m, up, down in zip(
        np.mean(x_loadings, axis=0)[:, args.mode],
        np.percentile(x_loadings, 97.5, axis=0)[:, args.mode],
        np.percentile(x_loadings, 2.5, axis=0)[:, args.mode])]
    probas = [p for m,p in sorted(zip(means, means_db))]
    ax.barh(range(len(labels)), probas, 0.9, color='gold', align='center')
    ax.set_ylim(-0.5, len(labels)-0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.1])
    ax.set_xticklabels([0, 0.1])
    ax.set_title(titles[i])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
axes[0].set_yticks(range(len(labels)))
axes[0].set_yticklabels(labels)
figfile = os.path.join(args.figure_folder, 'hist_loadings_comparison.png')
fig.savefig(figfile)
print(f'Figure saved at {figfile}')

exit(0)