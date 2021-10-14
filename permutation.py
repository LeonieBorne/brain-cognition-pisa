#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import functions.preprocessing
import functions.pls
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.impute import SimpleImputer
import os
import numpy as np
from sklearn.utils import shuffle

# resolve WARNINGS
# resolve default values FALSE (e.g. mean)
# test output save

# command line arguments
parser = argparse.ArgumentParser(
    description='Permutation tests for checking modes robustness.')
parser.add_argument("brain", help="Path to csv file with brain data.")
parser.add_argument("cognition", help="Path to csv file with cognition data.")
parser.add_argument("info", help="Path to csv file with additional information.")
parser.add_argument("-a", "--age_col", help="Column name in info csv with participants age", default="Age")
parser.add_argument("-s", "--sex_col", help="Column name in info csv with participants sex (1 if Male, 0 if Female)", default="Sex")
parser.add_argument("-g", "--group_col", help="Column name in info csv with participants clinical status (1 if healthy)", default="Group")
parser.add_argument("-r", "--regress_out_confounds", help="Regress out confounds.", action="store_false")
parser.add_argument("-m", "--mean", help="Average brain measurement (columns 'xxx_left' and 'xxx_right').", action="store_true")
parser.add_argument("-d", "--drop_rate", help="Maximum percentage of missing values without removing the subject/feature.", type=float, default=0.5)
parser.add_argument("-t", "--train_hc_only", help="Train only on healthy participants", action="store_true")
parser.add_argument("--cca", help="Use CCA instead of PLS model", action="store_false")
parser.add_argument("-p", "--nperm", help="Number of permutation tests", type=int, default=1000)
parser.add_argument("-f", "--figure_file", help="Output figure", default="")
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
    args.group_col, args.age_col, args.sex_col,
    args.drop_rate, args.regress_out_confounds,
    args.train_hc_only, args.mean, verbose=True)
    
# permutation tests
if args.cca:
    pipeline = functions.pls.PLSPipeline(CCA(n_components=5),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
    score_func = np.corrcoef
else:
    pipeline = functions.pls.PLSPipeline(PLSCanonical(n_components=5),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
    score_func = np.cov

x_scores, y_scores = pipeline.fit_transform(Xbrain, Ycogn)
n_comp = pipeline.PLS.n_components
ref_score = np.diag(score_func(
    x_scores, y_scores, rowvar=False)[:n_comp, n_comp:])

scores = []
for i in range(args.nperm):
    X = Xbrain
    Y = shuffle(Ycogn, random_state=i)
    x_scores, y_scores = pipeline.fit_transform(X, Y)
    scores.append(np.diag(score_func(
        x_scores, y_scores, rowvar=False)[:n_comp, n_comp:]))
scores = np.array(scores)

# print result
def print_pval(p):
    if p == 0:
        pstr = '<0.001'
    elif p == 1:
        pstr = '>0.99'
    elif p < 0.01:
        pstr = f'={p:.4f}'
    elif p < 0.1:
        pstr = f'={p:.3f}'
    else:
        pstr = f'={p:.2f}'
    return pstr

p = 0.05
scores = np.array(scores)
up_list = []
print()
pvals, zcov = [], []
for mode in range(n_comp):
    sc = ref_score[mode]
    zsc = (sc-np.mean(scores[:, 0]))/np.std(scores[:, 0])
    up = np.quantile(scores[:, 0], 1-p)
    up_list.append(up)
    pvals.append(sum(scores[:, 0] >= sc)/args.nperm) 
    zcov.append(zsc)
rstr = f'RESULT: 1st mode, p{print_pval(pvals[0])}'
if pvals[0] > 0.05:
    rstr += ')'
else:
    rstr += f', z={zcov[0]:.2f}; 2nd mode, p{print_pval(pvals[1])}'
print(rstr)

# save output csv file
if len(args.output_file) != 0:
    df = pd.DataFrame(scores.T)
    df.columns = [f'perm_{c}' for c in df.columns]
    df['score'] = ref_score
    df['zscore'] = zcov
    df.to_csv(args.output_file)

# figure
if len(args.figure_file) != 0:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 24})
    plt.bar(range(1, n_comp+1), up_list, color='silver', alpha=0.4)
    plt.plot(range(1, n_comp+1), ref_score, 'bo')
    plt.xlabel("modes")
    if args.cca:
        plt.ylabel("correlation")
    else:
        plt.ylabel("covariance")
    plt.title('Permutation tests')
    plt.xticks(range(1, n_comp+1))
    plt.violinplot(scores)
    plt.savefig(args.figure_file)

exit(0)