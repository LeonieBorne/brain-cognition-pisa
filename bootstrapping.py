#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import functions.preprocessing
import functions.pls
import functions.snapshots
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import os
import numpy as np
import matplotlib.pyplot as plt


# command line arguments
parser = argparse.ArgumentParser(
    description='Bootstrapping for checking the contribution of each individual score (a specific cognitive test or brain measurement) to the shared variance.')
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
parser.add_argument("-n", "--nboot", help="Number of bootstrapping tests", type=int, default=1000)
parser.add_argument("--modes", help="Number of modes", type=int, default=1)
# output arguments
parser.add_argument("-o", "--output_folder", help="Directory to output folder (if not specified, output csv file and figure(s) will be saved in the current folder)", default=os.getcwd())
parser.add_argument("--sulci_snapshot", help="Cortical sulci snapshots (only available if brain csv columns are sulci names)", action="store_true")

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
    
# bootstrapping
if args.cca:
    print('Using CCA...')
    pipeline = functions.pls.PLSPipeline(CCA(n_components=args.modes, max_iter=1000),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
else:
    print('Using Canonical PLS...')
    pipeline = functions.pls.PLSPipeline(PLSCanonical(n_components=args.modes, max_iter=1000),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

print(f'Running {args.nboot} bootstrapping tests...')
np.random.seed(0)
Xt, Yt = pipeline.fit_transform(Xbrain, Ycogn)
    
y_loadings, x_loadings = [], []
for i in range(args.nboot):
    Xr, Yr = resample(Xbrain, Ycogn)
    Xrt, Yrt = pipeline.fit_transform(Xr, Yr)
    y_loadings.append(pipeline.PLS.y_loadings_)
    x_loadings.append(pipeline.PLS.x_loadings_)
x_loadings = np.array(x_loadings)
y_loadings = np.array(y_loadings)

# print results
print()
for comp in range(pipeline.PLS.n_components):
    print(f'===== MODE {comp} =====')
    for mean_weight, std_weight, var in sorted(zip(
      np.mean(y_loadings, axis=0)[:, comp],
      np.std(y_loadings, axis=0)[:, comp], Ycogn.columns)):
        print(f'{mean_weight:.3f} +/- {std_weight:.3f} {var}')
print()
for comp in range(pipeline.PLS.n_components):
    print(f'===== MODE {comp} =====')
    for mean_weight, std_weight, var in sorted(zip(
      np.mean(x_loadings, axis=0)[:, comp],
      np.std(x_loadings, axis=0)[:, comp], Xbrain.columns)):
        print(f'{mean_weight:.3f} +/- {std_weight:.3f} {var}')
print()

# save output csv file
df = pd.DataFrame()
for loadings, columns in zip([x_loadings, y_loadings],
                             [Xbrain.columns, Ycogn.columns]):
    for mode in range(args.modes):
        for data, col in zip(loadings[:,:,mode].T, columns):
            df[f'mode{mode}_{col}'] = data
output_file = os.path.join(args.output_folder, 'bootstrapping_loadings.csv')
df.to_csv(output_file)
s = f'Bootstrapped loadings saved in {output_file}.\n'

# figure - cognitive loadings
for mode in range(args.modes):
    for loadings, columns, fname in zip([x_loadings[:,:,mode], y_loadings[:,:,mode]],
                                        [Xbrain.columns, Ycogn.columns],
                                        ['brain', 'cogn']):
        means = np.mean(loadings, axis=0)
        full_labels = [l for m, l in sorted(zip(means, columns))]
        scores = [m for m in sorted(means)]
        colors, labels = [], []
        for l in full_labels:
            if l.startswith('M.') or l.startswith('M_'):
                colors.append('lightcoral')
                labels.append(l[2:])
            elif l.startswith('L.') or l.startswith('L_'):
                colors.append('lightgreen')
                labels.append(l[2:])
            elif l.startswith('E.') or l.startswith('E_'):
                colors.append('lightblue')
                labels.append(l[2:])
            else:
                colors.append('plum')
                labels.append(l)
                
        plt.rcParams.update({'font.size': 40})
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax1 = ax.twinx()
        ax1.barh(range(len(labels)), scores, 0.9, color=colors, align='center',
             xerr=[[abs(m-v) for m, v in sorted(zip(means, np.percentile(loadings, 2.5, axis=0)), reverse=True)],
                   [abs(m-v) for m, v in sorted(zip(means, np.percentile(loadings, 97.5, axis=0)), reverse=True)]])
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        from matplotlib.patches import Patch
        if len(set(colors)) > 1:
            legend_elements = [Patch(facecolor='lightblue', label='Executive Functions'),
                               Patch(facecolor='lightcoral', label='Memory'),
                               Patch(facecolor='lightgreen', label='Language'),
                               Patch(facecolor='plum', label='Mood/Social Cognition')]
            ax1.legend(handles=legend_elements, loc='best')
        ax1.set_ylim(-0.5, len(labels)-0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()
        figure_file = os.path.join(
            args.output_folder, 
            f'boostrapping_figure_{fname}_loadings_mode{mode}.png')
        fig.savefig(figure_file)
        s += f'Figure with {fname} loadings for mode {mode} saved as {figure_file}.\n'

    if args.sulci_snapshot:
        # figure - brain loadings
        scores = [m if up*down > 0 else 0 for m, up, down in zip(
            np.mean(x_loadings, axis=0)[:, 0],
            np.percentile(x_loadings, 97.5, axis=0)[:, 0],
            np.percentile(x_loadings, 2.5, axis=0)[:, 0])]
        dict_sulcus = {s+'_left': x for s,x in zip(Xbrain.columns, scores) if x!=0}
        dict_reg = {0 : [0.5, -0.5, -0.5, 0.5], 1 : [0.5, 0.5, 0.5, 0.5]}
        for reg in [0, 1]:
            snapshot_file = os.path.join(
               args.output_folder, 
               f'bootstrapping_sulci_loadings_mode{mode}_view{reg}.png')
            functions.snapshots.view_sulcus_scores(
                dict_sulcus, side='left', reg_q=dict_reg[reg],
                minVal=0, maxVal=0.2, 
                snapshot=snapshot_file)
            s += f'Sulci snapshot (mode {mode}, view {reg}) saved as {snapshot_file}.\n'
print()
print(s)

exit(0)