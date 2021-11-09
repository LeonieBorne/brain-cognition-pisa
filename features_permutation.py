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
from scipy import stats 


# command line arguments
parser = argparse.ArgumentParser(
    description='Features permutation tests for checking age and clinical effects.')
parser.add_argument("brain", help="Path to csv file with brain data.")
parser.add_argument("cognition", help="Path to csv file with cognition data.")
parser.add_argument("info", help="Path to csv file with additional information, containing age ('Age' or age_col), sex ('Sex' or sex_col; 1 if Male, 0 if Female) and clinical status ('Group' or group_col; 1 if healthy).")
parser.add_argument("-a", "--age_col", help="Column name in info csv with participants age", default="Age")
parser.add_argument("-s", "--sex_col", help="Column name in info csv with participants sex (1 if Male, 0 if Female)", default="Sex")
parser.add_argument("-g", "--group_col", help="Column name in info csv with participants clinical status (1 if healthy)", default="Group")
# preprocessing options
parser.add_argument("-r", "--regress_out_confounds", help="Regress out confounds.", action="store_true")
parser.add_argument("-m", "--mean", help="Average brain measurement (columns 'xxx_left' and 'xxx_right').", action="store_true")
parser.add_argument("-d", "--drop_rate", help="Maximum percentage of missing values without removing the subject/feature.", type=float, default=0.5)
parser.add_argument("-t", "--train_hc_only", help="Train only on healthy participants", action="store_true")
# model options
parser.add_argument("--cca", help="Use CCA instead of PLS model", action="store_true")
parser.add_argument("-n", "--nperm", help="Number of permutation tests", type=int, default=1000)
parser.add_argument("--mode", help="Tested mode (1rst: 0; 2nd: 1; etc.)", type=int, default=0)
# output arguments
parser.add_argument("-o", "--output_folder", help="Directory to output folder (if not specified, output csv file and figure(s) will be saved in the current folder)", default=os.getcwd())

args = parser.parse_args()

# load data
df_brain, df_cogn, df_info = functions.preprocessing.check_csv_files(
    args.brain, args.cognition, args.info, 
    args.age_col, args.sex_col, args.group_col)

if args.drop_rate < 0 or args.drop_rate > 1:
    print(f'Drop rate should be between 0 and 1. {args.drop_rate} is not a valid value.')
    exit(1)
    
# preprocessing
df_brain, df_cogn, df_info, Xbrain, Ycogn = functions.preprocessing.preprocessing(
    df_brain, df_cogn, df_info, 
    group_col=args.group_col, age_col=args.age_col, sex_col=args.sex_col,
    drop_rate=args.drop_rate, regress_out_confounds=args.regress_out_confounds,
    train_hc_only=args.train_hc_only, mean=args.mean, verbose=True)
    
# features permutation
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

def print_pval(p):
    if p == 0:
        pstr = '<0.001'
    elif p == 1:
        pstr = '>0.99'
    elif p < 0.001:
        pstr = f'={p:.1e}'
    elif p < 0.01:
        pstr = f'={p:.4f}'
    elif p < 0.1:
        pstr = f'={p:.3f}'
    else:
        pstr = f'={p:.2f}'
    return pstr

pipeline.fit(Xbrain, Ycogn)
Xbrain_r, Ycogn_r = pipeline.transform(df_brain, df_cogn)

np.random.seed(0)
nperm = 1000
Xbrain_perm = np.zeros([nperm, len(Xbrain_r)])
Ycogn_perm = np.zeros([nperm, len(Ycogn_r)])
for perm in range(args.nperm):
    X = df_brain
    Y = df_cogn
    if pipeline.Ximputer is not None:
        X = pipeline.Ximputer.transform(X)
    if pipeline.Yimputer is not None:
        Y = pipeline.Yimputer.transform(Y)

    # Scaling
    X = pipeline.Xscaler.transform(X)
    Y = pipeline.Yscaler.transform(Y)

    # Permutation
    np.random.shuffle(np.transpose(X))
    np.random.shuffle(np.transpose(Y))
    
    Xperm, Yperm = pipeline.PLS.transform(X, Y)
    Xbrain_perm[perm] = Xperm[:, args.mode]
    Ycogn_perm[perm] = Yperm[:, args.mode]

# save output file
df = pd.DataFrame()
for data, pname in zip([Xbrain_perm, Ycogn_perm],
                       ['brain', 'cogn']):
    for perm in range(args.nperm):
        df[f'{pname}_perm{perm}'] = data[perm]
output_file = os.path.join(args.output_folder, 'features_permutation_projections.csv')
df.to_csv(output_file)
print(f'\nPermuted projections saved in {output_file}.')
    
# figure
plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(4, 1, figsize=(9, 24))
nperm = len(Xbrain_perm)
groups = sorted(list(set(df_info[args.group_col])))
ngroup = len(groups)
for row in range(4):
    ref_score = np.zeros(ngroup)
    scores = np.zeros([len(Xbrain_perm), ngroup])
    pvals = []
    for i in range(ngroup):
        if row == 3:
            ref_score[i] = np.mean(Xbrain_r[df_info[args.group_col] == groups[i],args.mode])
            for perm in range(nperm):
                scores[perm, i] = np.mean(Xbrain_perm[perm, df_info[args.group_col] == groups[i]])
        elif row == 2:
            ref_score[i] = np.mean(Ycogn_r[df_info[args.group_col] == groups[i],args.mode])
            for perm in range(nperm):
                scores[perm, i] = np.mean(Ycogn_perm[perm, df_info[args.group_col] == groups[i]])
        elif row == 1:
            linreg = stats.linregress(df_info.loc[df_info[args.group_col] == groups[i], 'Age'], 
                                      Xbrain_r[df_info[args.group_col] == groups[i],args.mode])
            ref_score[i] = linreg.slope
            for perm in range(nperm):
                linreg = stats.linregress(df_info.loc[df_info[args.group_col] == groups[i], 'Age'], 
                                          Xbrain_perm[perm, df_info[args.group_col] == groups[i]])
                scores[perm, i] = linreg.slope
        else:
            linreg = stats.linregress(df_info.loc[df_info[args.group_col] == groups[i], 'Age'], 
                                      Ycogn_r[df_info[args.group_col] == groups[i],args.mode])
            ref_score[i] = linreg.slope
            for perm in range(args.nperm):
                linreg = stats.linregress(df_info.loc[df_info[args.group_col] == groups[i], 'Age'], 
                                          Ycogn_perm[perm, df_info[args.group_col] == groups[i]])
                scores[perm, i] = linreg.slope
        pvals.append((sum(scores[:,i] >= ref_score[i]))/nperm)
    ax = axes[row]
    for perm in range(100):
        ax.plot(range(1, ngroup+1), scores[perm], color='silver', alpha=1, 
                linewidth=0.5, zorder=1)
    color = 'salmon'
    ax.plot(range(1, ngroup+1), ref_score, color=color, alpha=1, linewidth=3, zorder=3)
    ax.plot(range(1, ngroup+1), ref_score, color=color, marker='o', markersize=10, zorder=3)
    ax.set_xticks(range(1, ngroup+1))
    groups_name = ['HC', 'MCI', 'AD']
    xticklabels = [f'{groups_name[i]}\np{print_pval(pvals[i])}' for i in range(ngroup)]
    if row in [2, 3]:
        xticklabels[0] = 'HC'
    ax.set_xticklabels(xticklabels)
    v = ax.violinplot(scores, showmedians=True)
    for pc in v['bodies']:
        pc.set_zorder(2)
        pc.set_color('dimgrey')
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = v[partname]
        vp.set_edgecolor('dimgrey')
axes[0].yaxis.labelpad = 20
axes[0].set_ylabel('(a) age-effect\non cognitive projections')
axes[1].yaxis.labelpad = 20
axes[1].set_ylabel('(b) age-effect\non SW projections')
axes[2].yaxis.labelpad = 20
axes[2].set_ylabel('(c) mean\ncognitive projections')
axes[3].yaxis.labelpad = 20
axes[3].set_ylabel('(d) mean\nSW projections')
# axes[0].set_yticks([0,0.1,0.2,0.3])
# axes[1].set_yticks([0.1,0.2,0.3,0.4])
# axes[2].set_yticks([0,5,10])
# axes[3].set_yticks([0,5,10])
fig.tight_layout()
figure_file = os.path.join(args.output_folder, 'features_permutation_figure.png')
fig.savefig(figure_file)
print(f'Figure saved as {figure_file}.')

exit(0)