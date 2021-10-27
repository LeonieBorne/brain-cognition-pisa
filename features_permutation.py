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
    description='Permutation tests for checking modes robustness.')
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
parser.add_argument("-f", "--figure_file", help="Output figure file", default="")
parser.add_argument("-o", "--output_file", help="Output csv file", default="")

args = parser.parse_args()

# load data
if os.path.exists(args.brain):
    df_brain = pd.read_csv(args.brain, index_col=0)
else:
    print(f'Brain csv file not found: {args.brain}')
    exit(1)
if os.path.exists(args.cognition):
    df_cogn = pd.read_csv(args.cognition, index_col=0)
else:
    print(f'Cognition csv file not found: {args.cognition}')
    exit(1)
if os.path.exists(args.info):
    df_info = pd.read_csv(args.info, index_col=0)
else:
    print(f'Csv file containing the additional information not found: {args.info}.')
    exit(1)

# check arguments
if args.group_col not in df_info.columns:
    print(f'Column {args.group_col} not in {args.info}.')
    exit(1)
if args.age_col not in df_info.columns:
    print(f'Column {args.age_col} not in {args.info}.')
    exit(1)
if args.sex_col not in df_info.columns:
    print(f'Column {args.sex_col} not in {args.info}.')
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
if len(args.output_file) != 0:
    df = pd.DataFrame()
    for data, pname in zip([Xbrain_perm, Ycogn_perm],
                           ['brain', 'cogn']):
        for perm in range(args.nperm):
            df[f'{pname}_perm{perm}'] = data[perm]
    df.to_csv(args.output_file)
    print(f'Results saved in {args.output_file}.')
    
# figure
if len(args.figure_file) != 0:
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(4, 1, figsize=(9, 24))
    nperm = len(Xbrain_perm)
    for row in range(4):
        ref_score = np.zeros(3)
        scores = np.zeros([len(Xbrain_perm), 3])
        pvals = []
        for g in range(3):
            if row == 3:
                ref_score[g] = np.mean(Xbrain_r[df_info[args.group_col] == g+1,args.mode])
                for perm in range(nperm):
                    scores[perm, g] = np.mean(Xbrain_perm[perm, df_info[args.group_col] == g+1])
            elif row == 2:
                ref_score[g] = np.mean(Ycogn_r[df_info[args.group_col] == g+1,args.mode])
                for perm in range(nperm):
                    scores[perm, g] = np.mean(Ycogn_perm[perm, df_info[args.group_col] == g+1])
            elif row == 1:
                linreg = stats.linregress(df_info.loc[df_info[args.group_col] == g+1, 'Age'], 
                                          Xbrain_r[df_info[args.group_col] == g+1,args.mode])
                ref_score[g] = linreg.slope
                for perm in range(nperm):
                    linreg = stats.linregress(df_info.loc[df_info[args.group_col] == g+1, 'Age'], 
                                              Xbrain_perm[perm, df_info[args.group_col] == g+1])
                    scores[perm, g] = linreg.slope
            else:
                linreg = stats.linregress(df_info.loc[df_info[args.group_col] == g+1, 'Age'], 
                                          Ycogn_r[df_info[args.group_col] == g+1,args.mode])
                ref_score[g] = linreg.slope
                for perm in range(args.nperm):
                    linreg = stats.linregress(df_info.loc[df_info[args.group_col] == g+1, 'Age'], 
                                              Ycogn_perm[perm, df_info[args.group_col] == g+1])
                    scores[perm, g] = linreg.slope
            pvals.append((sum(scores[:,g] >= ref_score[g]))/nperm)
        ax = axes[row]
        for perm in range(100):
            ax.plot([1,2,3], scores[perm], color='silver', alpha=1, 
                    linewidth=0.5, zorder=1)
        color = 'salmon'
        ax.plot([1,2,3], ref_score, color=color, alpha=1, linewidth=3, zorder=3)
        ax.plot(range(1, 3+1), ref_score, color=color, marker='o', markersize=10, zorder=3)
        ax.set_xticks(range(1, 3+1))
        xticklabels = [f'HC\np{print_pval(pvals[0])}',
                        f'MCI\np{print_pval(pvals[1])}', 
                        f'AD\np{print_pval(pvals[2])}']
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
    axes[0].set_yticks([0,0.1,0.2,0.3])
    axes[1].set_yticks([0.1,0.2,0.3,0.4])
    axes[2].set_yticks([0,5,10])
    axes[3].set_yticks([0,5,10])
    fig.tight_layout()
    fig.savefig(args.figure_file)

exit(0)