#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import functions.preprocessing
import functions.pls
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle


# command line arguments
parser = argparse.ArgumentParser(
    description='Permutation tests for checking modes robustness.')
parser.add_argument("brain", help="Path to csv file with brain data")
parser.add_argument("cognition", help="Path to csv file with cognition data")
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
parser.add_argument("--cca", help="Use CCA instead of Canonical PLS model", action="store_true")
parser.add_argument("-p", "--nperm", help="Number of permutation tests", type=int, default=1000)
# output arguments
parser.add_argument("-f", "--figure_file", help="Output figure", default="")
parser.add_argument("-o", "--output_file", help="Output csv file", default="")

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
    
# permutation tests
if args.cca:
    print('Using CCA...')
    pipeline = functions.pls.PLSPipeline(CCA(n_components=5, max_iter=1000),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
    score_func = np.corrcoef
else:
    print('Using Canonical PLS...')
    pipeline = functions.pls.PLSPipeline(PLSCanonical(n_components=5, max_iter=1000),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
    score_func = np.cov

print(f'Running {args.nperm} permutation tests...')
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
    elif p < 0.001:
        pstr = f'={p:.1e}'
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
if pvals[0] <= 0.05:
    rstr += f', z={zcov[0]:.2f}; 2nd mode, p{print_pval(pvals[1])}'
print(rstr)

# save output csv file
if len(args.output_file) != 0:
    df = pd.DataFrame(scores.T)
    df.columns = [f'perm_{c}' for c in df.columns]
    df['score'] = ref_score
    df['zscore'] = zcov
    df.to_csv(args.output_file)
    print(f'Results saved in {args.output_file}.')

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
    print(f'Figure saved as {args.figure_file}.')

exit(0)