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
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn import svm


# command line arguments
parser = argparse.ArgumentParser(
    description='Clinical status classification based on brain and/or cognitive projections.')
parser.add_argument("brain", help="Path to csv file with brain data.")
parser.add_argument("cognition", help="Path to csv file with cognition data.")
parser.add_argument("info", help="Path to csv file with additional information, containing age ('Age' or age_col), sex ('Sex' or sex_col; 1 if Male, 0 if Female) and clinical status ('Group' or group_col; 1 if healthy).")
parser.add_argument("-a", "--age_col", help="Column name in info csv with participants age", default="Age")
parser.add_argument("-s", "--sex_col", help="Column name in info csv with participants sex (1 if Male, 0 if Female)", default="Sex")
parser.add_argument("-g", "--group_col", help="Column name in info csv with participants clinical status (1 if healthy)", default="Group")
parser.add_argument("--brain2", help="Path to csv file with the second brain data.", default="")
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
parser.add_argument("-f", "--figure_folder", help="Output figure folder", default="")
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
if len(args.brain2) != 0:
    if os.path.exists(args.brain2):
        df_brain2 = pd.read_csv(args.brain2, index_col=0)
    else:
        print(f'Brain2 csv file not found: {args.brain2}')
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

# projection computation function
def compute_lv(df_brain, df_cogn, df_info, pipeline, train=None):
    dfb, dfc, dfi, Xbrain, Ycogn = functions.preprocessing.preprocessing(
        df_brain, df_cogn, df_info, 
        group_col=args.group_col, age_col=args.age_col, sex_col=args.sex_col,
        drop_rate=args.drop_rate, regress_out_confounds=args.regress_out_confounds,
        train_hc_only=args.train_hc_only, mean=args.mean, verbose=False)
    
    if train is None:
        pipeline.fit(Xbrain, Ycogn)
    else:
        df_train = dfi.iloc[train]
        df_train_hc = df_train[df_train[args.group_col] == 1]
        pipeline.fit(Xbrain.loc[df_train_hc.index], Ycogn.loc[df_train_hc.index])
    Xsulci_r, Ycogn_r = pipeline.transform(dfb, dfc)
    return Xsulci_r[:,args.mode], Ycogn_r[:,args.mode], dfi[args.group_col]

# figure functions
def scatter_plot(lv1_name='LV1', lv2_name='LV2', comparison=False):
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.8
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    # start with a square Figure
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(6, 6))    
    ax = fig.add_axes(rect_scatter)
    # ax.set_yticks([-8,-4,0,4])
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histx.axis('off')
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histy.axis('off')
    # main scatter plot
    col={1:'steelblue', 2:'tab:orange', 3:'red'}
    dict_label={1:'HC', 2:'MCI', 3:'AD'}
    if comparison:
        lv1, _, gr = compute_lv(df_brain, df_cogn, df_info, pipeline)
        lv2, _, _ = compute_lv(df_brain2, df_cogn, df_info, pipeline)
    else:
        lv1, lv2, gr = compute_lv(df_brain, df_cogn, df_info, pipeline)
    for i in set(gr):
        ax.scatter(lv1[gr == i], lv2[gr == i],
                   label=dict_label[i], marker="o", c=col[i], s=75)
    ax.legend()
    ax.set_xlabel(lv1_name)
    ax.set_ylabel(lv2_name)    
    # histogram
    for i in set(gr):
        h2 = sns.kdeplot(y = lv2[gr == i],
                         fill = True, linewidth = 3, 
                         ax = ax_histy, color=col[i])
        h1 = sns.kdeplot(x = lv1[gr == i],
                         shade = True, linewidth = 3, 
                         ax = ax_histx, color=col[i])
    figfile = os.path.join(args.figure_folder, f'{lv1_name}_vs_{lv2_name}_scatter_plot.png')
    fig.savefig(figfile, format='png', bbox_extra_artists=(h1,h2), bbox_inches='tight')
    print(f'Figure saved at {figfile}')

def auc_plot(lv1_name='LV1', lv2_name='LV2', comparison=False): 
    lw=3
    if comparison:
        lv1, _, gr = compute_lv(df_brain, df_cogn, df_info, pipeline)
        lv2, _, _ = compute_lv(df_brain2, df_cogn, df_info, pipeline)
    else:
        lv1, lv2, gr = compute_lv(df_brain, df_cogn, df_info, pipeline)
    y = np.array([0 if g==1 else 1 for g in gr])
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), sharex=True, sharey=True)
    for X, title, ax in zip([np.array([lv1]).T, np.array([lv2]).T, 
                             np.array([lv1, lv2]).T],
                            [lv1_name, lv2_name, f'{lv1_name} & {lv2_name}'], axes):
        random_state = np.random.RandomState(0)
        cv = StratifiedKFold(n_splits=10)
        classifier = svm.SVC(kernel='linear', probability=True,
                             random_state=random_state)
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for i, (train, test) in enumerate(cv.split(X, y)):
            if comparison:
                train_lv1, _, _ = compute_lv(
                    df_brain, df_cogn, df_info, pipeline, train)
                train_lv2, _, _ = compute_lv(
                    df_brain2, df_cogn, df_info, pipeline, train)
            else:
                train_lv1, train_lv2, _ = compute_lv(
                    df_brain, df_cogn, df_info, pipeline, train)
            if title == lv1_name:
                X = np.array([train_lv1]).T
            elif title == lv2_name:
                X = np.array([train_lv2]).T
            else:
                X = np.array([train_lv1, train_lv2]).T
            classifier.fit(X[train], y[train])
            viz = plot_roc_curve(classifier, X[test], y[test],
                                  alpha=0, lw=lw, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        
        ax.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',
                label='Chance', alpha=.8)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=lw, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.get_legend().remove()
        ax.set_title(f'{title}\n(AUC={mean_auc:.2f} std. {std_auc:.2f})')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([0,0.5,1])
    
    axes[1].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    figfile = os.path.join(args.figure_folder, f'{lv1_name}_vs_{lv2_name}_AUC_plot.png')
    fig.tight_layout()
    fig.savefig(figfile, format='png')
    print(f'Figure saved at {figfile}')

# figure - classification
scatter_plot(lv1_name='Brain', lv2_name='Cognition', comparison=False)
auc_plot(lv1_name='Brain', lv2_name='Cognition', comparison=False)

# figure - classification comparison brain 1/2
if len(args.brain2) != 0:
    scatter_plot(lv1_name='Brain1', lv2_name='Brain2', comparison=True)
    auc_plot(lv1_name='Brain1', lv2_name='Brain2', comparison=True)

exit(0)