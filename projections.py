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
from pingouin import ancova


# command line arguments
parser = argparse.ArgumentParser(
    description='Save projections values and/or plot age effect.')
parser.add_argument("brain", help="Path to csv file with brain data.")
parser.add_argument("cognition", help="Path to csv file with cognition data.")
parser.add_argument("info", help="Path to csv file with additional information, containing age ('Age' or age_col), sex ('Sex' or sex_col; 1 if Male, 0 if Female) and clinical status ('Group' or group_col; 1 if healthy).")
parser.add_argument("-a", "--age_col", help="Column name in info csv with participants age", default="Age")
parser.add_argument("-s", "--sex_col", help="Column name in info csv with participants sex (1 if Male, 0 if Female)", default="Sex")
parser.add_argument("-g", "--group_col", help="Column name in info csv with participants clinical status (1 if healthy)", default="Group")
parser.add_argument("--amyloid_col", help="Column name in info csv with participants amyloid status ('Positive' or 'Negative')", default="Amyloid")
parser.add_argument("--prs_col", help="Column name in info csv with participants PRS score", default="PRS_noAPOE")
parser.add_argument("--apoe_col", help="Column name in info csv with participants APOE status (4 if ε4-carrier else 0)", default="APOEe4")
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
parser.add_argument("-f", "--figure_folder", help="Output figures folder", default="")
parser.add_argument("-o", "--output_file", help="Output csv file", default="")
parser.add_argument("--plot_age_effect", help="Plot age effect", action="store_true")

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
    
# compute projections
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

pipeline.fit(Xbrain, Ycogn)
Xbrain_r, Ycogn_r = pipeline.transform(df_brain, df_cogn)

# save projections
if len(args.output_file) != 0:
    df = pd.DataFrame()
    for data, pname in zip([Xbrain_r, Ycogn_r],
                           ['brain', 'cogn']):
        for mode in range(args.modes):
            df[f'mode{mode}_{pname}'] = data[:,mode]
    df.to_csv(args.output_file)
    print(f'Results saved in {args.output_file}.')

# plot functions definition
def plot_age_effect(X, Y, df_info):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(4, 4, figsize=(30, 20), sharey='row',
                             gridspec_kw={'width_ratios': [4, 2, 4, 2]})
    # sex
    print('------ SEX')
    col={0:'salmon', 1:'lightseagreen'}
    dict_label={0:'Female', 1:'Male'}
    covar = [args.age_col]
    axes = plot_comparison(X[df_info[args.group_col] == 1], Y[df_info[args.group_col] == 1], 
                           df_info[df_info[args.group_col] == 1], args.sex_col, covar,
                           axes, 0, dict_label, col, mode=0)
    axes[0, 0].legend(loc='upper left')
    axes[0, 2].set_title('Sulcal Width')
    axes[0, 0].set_title('Cognitive Scores')
    axes[0, 0].set_ylabel('(a)', rotation=0, fontsize=40)
    axes[0, 0].yaxis.labelpad = 30
    # groups
    print('------ GROUPS')
    col={1:'steelblue', 2:'tab:orange', 3:'red'}
    dict_label={1:'HC', 2:'MCI', 3:'AD'}
    covar = [args.age_col, args.sex_col]
    axes = plot_comparison(X, Y, df_info, args.group_col, covar, 
                           axes, 1, dict_label, col, mode=0)
    axes[1, 0].legend(loc='lower left', ncol=3)
    axes[1, 0].set_ylabel('(b)', rotation=0, fontsize=40)
    axes[1, 0].yaxis.labelpad = 30
    # amyloid
    print('------ AMYLOID')
    col={'Negative':'sandybrown', 'Positive':'darkseagreen'}
    dict_label={'Negative':'Aβ negative', 'Positive':'Aβ positive'}
    covar = [args.age_col, args.sex_col]
    axes = plot_comparison(X[df_info[args.group_col] == 1],
                           Y[df_info[args.group_col] == 1], 
                           df_info[df_info[args.group_col] == 1], 
                           args.amyloid_col, covar,
                           axes, 2, dict_label, col, mode=0)
    axes[2, 0].legend(loc='upper left')
    axes[2, 0].set_ylabel('(c)', rotation=0, fontsize=40)
    axes[2, 0].yaxis.labelpad = 30
    # apoe
    print('------ APOE')
    col={0:'orchid', 4:'lightskyblue'}
    dict_label={0:'no APOE ε4', 4:'APOE ε4'}
    colname = args.apoe_col
    covar = [args.age_col, args.sex_col]
    if args.prs_col in df_info.columns:
        covar = covar + [args.prs_col]
    if 'Twin' in df_info.columns:
        df_twin = df_info[df_info.Twin == 'MZ']
        twin_list = []
        for idx in df_twin.index:
            if not df_twin.loc[idx, 'Twin_ID'] in twin_list:
                twin_list.append(idx)
        rm_list = [idx for idx in df_twin.index if idx not in twin_list]
        select = np.array([True if row[args.group_col] == 1 and idx not in rm_list else False for idx, row in df_info.iterrows()])
    else:
        select = np.array(df_info[args.group_col] == 1)
    axes = plot_comparison(X[select], Y[select], 
                           df_info[select], colname, covar,
                           axes, 3, dict_label, col, mode=0)
    axes[3, 0].legend(loc='upper left')
    axes[3, 0].set_xlabel("age")
    axes[3, 2].set_xlabel("age")
    axes[3, 0].yaxis.labelpad = 30
    axes[3, 0].set_ylabel('(d)', rotation=0, fontsize=40)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.tight_layout()
    return fig

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

def plot_comparison(X, Y, df_info, between, covar, axes, line,
                    dict_label, col, mode=0):
    groups = df_info[between]
    lv = 'Brain'
    s = 25 if len(df_info)>500 else 75
    pvals = []
    for ax, axh, x in zip([axes[line, 2], axes[line, 0]], 
                          [axes[line, 3], axes[line, 1]], 
                          [X, Y]):
        x = np.array(x)[~pd.isnull(groups)]
        y = np.array(df_info.Age)[~pd.isnull(groups)]
        gr = np.array(groups)[~pd.isnull(groups)]
        grs = list(set(gr))
        for i in set(gr):
            path = ax.scatter(y[gr == i], x[gr == i],
                              label=dict_label[i], marker="o",
                              c=col[i], s=s)
            Xg, Yg = x[gr == i], y[gr == i]
            linreg = stats.linregress(Yg[~np.isnan(Yg)],
                                      Xg[~np.isnan(Yg)])
            ax.plot(y, linreg.intercept + linreg.slope*y, 
                    color=path.get_facecolors()[0].tolist(),
                    linewidth=4)
        data = [list(x[gr == g]) for g in sorted(list(set(gr)))]
        labels = [dict_label[g] for g in sorted(list(set(gr)))]
        colors = [col[g] for g in sorted(list(set(gr)))]
        bplot = axh.boxplot(data, notch=True, vert=True, widths=0.3,
                            patch_artist=True, whis=100, labels=labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        df = df_info[covar + [between]][~pd.isnull(groups)]
        df['dv'] = x
        if len(set(gr)) > 2:
            # test mean difference
            for g, th in zip([[1, 2], [2, 3], [1, 3]], [0, 5, 15]):
                anc = ancova(data=df[df_info[args.group_col].isin(g)], dv='dv',
                             covar=covar, between=between)
                print(f'{lv}: mean difference {dict_label[g[0]]} vs. {dict_label[g[1]]} p{print_pval(anc.loc[0, "p-unc"])}')
                if anc.loc[0, 'p-unc'] < 0.05:
                    star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
                    axh.plot([g[0] + 0.25, g[0] + 0.25, g[1] - 0.25, g[1] - 0.75],
                              [th, th, th, th], linewidth=2, color='k')
                    axh.text(sum(g)/2, th+1, star, ha='center', va='bottom', color='k')
            # test slope difference
            for grs in [[1, 2], [2, 3], [1, 3]]:
                tstat, pval = test_slopes_anova(y[gr==grs[0]], x[gr==grs[0]], 
                                          y[gr==grs[1]], x[gr==grs[1]])
                print(f'{lv}: slope difference {dict_label[grs[0]]} vs. {dict_label[grs[1]]} p{print_pval(pval)}')
        else:
            grs = list(set(gr))
            # test mean difference
            anc = ancova(data=df, dv='dv', covar=covar, between=between)
            print(f'{lv}: mean difference p{print_pval(anc.loc[0, "p-unc"])}')
            if anc.loc[0, 'p-unc'] < 0.05:
                star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
                axh.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
                axh.text(1.5, 1, star, ha='center', va='bottom', color='k')
            # test slope difference
            tstat, pval = test_slopes_anova(y[gr==grs[0]], x[gr==grs[0]], 
                                            y[gr==grs[1]], x[gr==grs[1]])
            print(f'{lv}: slope difference p{print_pval(pval)}')
        lv = 'Cognition'
    return axes

def test_slopes_anova(X1, Y1, X2, Y2):
    df1 = pd.DataFrame({'x':X1, 'y':Y1})
    df1['group'] = 1
    df2 = pd.DataFrame({'x':X2, 'y':Y2})
    df2['group'] = 2
    df = pd.concat([df1, df2], ignore_index=True)
    df['interaction'] = [df.loc[idx, 'x'] if df.loc[idx, 'group'] == 1 else 0 for idx in df.index]
    anc = ancova(dv='y', covar=['x', 'interaction'], between='group', data=df)
    anc.index = anc.Source
    return anc.loc['interaction', 'F'], anc.loc['interaction', 'p-unc']

# save figures
if args.plot_age_effect:
    print()
    for mode in range(args.modes):
        print(f'======== MODE {mode} ========')
        fig = plot_age_effect(Xbrain_r[:, mode], Ycogn_r[:, mode], df_info)
        fig.savefig(os.path.join(args.figure_folder, 
                                 f'age_plot_mode{mode}.png'))
        print()
    print(f'Figures saved in {args.figure_folder}.')

exit(0)