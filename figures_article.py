#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats 
from pingouin import ancova
from collections import Counter
import seaborn as sns

from sklearn.cross_decomposition import PLSCanonical
from sklearn.impute import SimpleImputer

from uon.preprocessing import preprocessing
import uon.pls as pls
from uon.snapshots import view_sulcus_scores


plt.style.use('default')


database = 'pisa'
regress_out_confounds = False
train_hc_only = True
measure='opening'
# measure='GM_thickness'

database = 'aibl'
regress_out_confounds = False
train_hc_only = True
measure='opening'
# measure='GM_thickness'

database = 'adni'
regress_out_confounds = False
train_hc_only = True
# measure='opening'
measure='GM_thickness'

# database = 'pisa'
# regress_out_confounds = True
# train_hc_only = False
# measure='opening'
# # measure='GM_thickness'

#############
# LOAD DATA #
#############

data_path = f'/home/leonie/Documents/data/{database}'
df_sulci = pd.read_csv(f'{data_path}/sMRI/sulci_morphometry_measurements/per_measurement/{measure}.csv', index_col=0)
df_cogn = pd.read_csv(f'{data_path}/cognitive_scores/neuropsych.csv', index_col=0)
df_gen = pd.read_csv(f'{data_path}/cognitive_scores/demographics.csv', index_col=0)

df_sulci, df_cogn, df_gen, Xsulci, Ycogn = preprocessing(
    df_sulci, df_cogn, df_gen, regress_out_confounds, train_hc_only)


####################
# PERMUTATION TEST #
####################

pipeline = pls.PLSPipeline(PLSCanonical(n_components=2),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
pls.permutation_test(Xsulci, Ycogn, pipeline)

################
# PLS LOADINGS #
################

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))
x_weights, y_weights, x_loadings, y_loadings = pls.bootstrapping(Xsulci, Ycogn, pipeline)

dict_cogn = {c:c[2:] for c in df_cogn.columns}
mode = 0

# cognitive weights
full_labels = Ycogn.columns
means = [m if up*down > 0 else 0 for m, up, down in zip(
    np.mean(y_loadings, axis=0)[:, mode],
    np.percentile(y_loadings, 97.5, axis=0)[:, mode],
    np.percentile(y_loadings, 2.5, axis=0)[:, mode])]
full_labels = Ycogn.columns
full_labels = [l for m, l in sorted(zip(means, full_labels)) if m != 0]
labels = [dict_cogn[s] if s in dict_cogn.keys() else s for s in full_labels] 
probas = [ m for m in sorted(means) if m != 0]
colors = []
for l in full_labels:
    if l.startswith('M.'):
        colors.append('lightcoral')
    elif l.startswith('L.'):
        colors.append('lightgreen')
    elif l.startswith('E.'):
        colors.append('lightblue')
    else:
        colors.append('plum')
        
plt.rcParams.update({'font.size': 40})
if database == 'pisa':
    fig, ax = plt.subplots(figsize=(20, 25))
else:
    fig, ax = plt.subplots(figsize=(20, 30))
ax.set_yticklabels([])
ax.set_yticks([])
ax1 = ax.twinx()
ax1.barh(range(len(labels)), probas, 0.9, color=colors, align='center')
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', label='Executive Functions'),
                   Patch(facecolor='lightcoral', label='Memory'),
                   Patch(facecolor='lightgreen', label='Language'),
                   Patch(facecolor='plum', label='Other')]
ax1.legend(handles=legend_elements, loc='best')
if measure == 'opening':
    ax1.set_xticks([0, -0.1, -0.2, -0.3])
else:
    ax1.set_xticks([0, 0.1, 0.2, 0.3])
ax1.set_ylim(-0.5, len(labels)-0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# sulcal weights
scores = [m if up*down > 0 else 0 for m, up, down in zip(
    np.mean(x_loadings, axis=0)[:, 0],
    np.percentile(x_loadings, 97.5, axis=0)[:, 0],
    np.percentile(x_loadings, 2.5, axis=0)[:, 0])]
dict_sulcus = {s+'_left': x for s,x in zip(Xsulci.columns, scores) if x!=0}
dict_reg = {0 : [0.5, -0.5, -0.5, 0.5], 1 : [0.5, 0.5, 0.5, 0.5]}
for side in ['left']:
    for reg in [0, 1]:
        view_sulcus_scores(
            dict_sulcus,
            side=side,
            reg_q=dict_reg[reg],
            minVal=0, maxVal=0.2,
            # background=[0,0,0,1],
            snapshot=f'/tmp/{side}{reg}_bootstrap_mode{mode}.png')

##############
# AGE EFFECT #
##############

def plot_age_effect(Xsulci_r, Ycogn_r, df_gen):
    plt.style.use('default')
    plt.rcParams.update({'font.size': 25})
    fig, axes = plt.subplots(4, 4, figsize=(30, 20), sharey='row',
                             gridspec_kw={'width_ratios': [4, 2, 4, 2]})
    # groups
    print('======== GROUPS ========')
    col={1:'steelblue', 2:'tab:orange', 3:'red'}
    dict_label={1:'HC', 2:'MCI', 3:'AD'}
    covar = ['Age', 'Sex']
    axes = plot_comparison(Xsulci_r, Ycogn_r, df_gen, 'Group', covar, 
                           axes, 0, dict_label, col, mode=0)
    axes[0, 0].legend(loc='lower left', ncol=3)
    axes[0, 2].set_title('Sulcal Width')
    axes[0, 0].set_title('Cognitive Scores')
    print()
    # sex
    print('======== SEX ========')
    col={0:'salmon', 1:'lightseagreen'}
    dict_label={0:'Female', 1:'Male'}
    covar = ['Age']
    axes = plot_comparison(Xsulci_r[df_gen.Group == 1], Ycogn_r[df_gen.Group == 1], 
                           df_gen[df_gen.Group == 1], 'Sex', covar,
                           axes, 1, dict_label, col, mode=0)
    axes[1, 0].legend(loc='upper left')
    print()
    # amyloid
    print('======== AMYLOID ========')
    col={'Negative':'sandybrown', 'Positive':'darkseagreen'}
    dict_label={'Negative':'Aβ negative', 'Positive':'Aβ positive'}
    covar = ['Age', 'Sex']
    axes = plot_comparison(Xsulci_r[df_gen.Group == 1], Ycogn_r[df_gen.Group == 1], 
                            df_gen[df_gen.Group == 1], 'Amyloid', covar,
                            axes, 2, dict_label, col, mode=0)
    axes[2, 0].legend(loc='upper left')
    print()
    # apoe
    print('======== APOE ========')
    col={0:'orchid', 4:'lightskyblue'}
    dict_label={0:'no APOE ε4', 4:'APOE ε4'}
    covar = ['Age', 'Sex']
    if 'PRS_noAPOE' in df_gen.columns:
        covar = covar + ['PRS_noAPOE']
    if 'Twin' in df_gen.columns:
        df_twin = df_gen[df_gen.Twin == 'MZ']
        twin_list = []
        for idx in df_twin.index:
            if not df_twin.loc[idx, 'Twin_ID'] in twin_list:
                twin_list.append(idx)
        rm_list = [idx for idx in df_twin.index if idx not in twin_list]
        select = np.array([True if row['Group'] == 1 and idx not in rm_list else False for idx, row in df_gen.iterrows()])
    else:
        select = np.array(df_gen.Group == 1)
    axes = plot_comparison(Xsulci_r[select], Ycogn_r[select], 
                            df_gen[select], 'APOEe4', covar,
                            axes, 3, dict_label, col, mode=0)
    axes[3, 0].legend(loc='upper left')
    axes[3, 0].set_xlabel("age")
    axes[3, 2].set_xlabel("age")
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel("latent variable")
    plt.tight_layout()
    
def plot_comparison(Xsulci_r, Ycogn_r, df_gen, between, covar, axes, line,
                    dict_label, col, mode=0):
    groups = df_gen[between]
    LV = 'SW'
    s = 25 if len(df_gen)>500 else 75
    pvals = []
    for ax, axh, x in zip([axes[line, 2], axes[line, 0]], 
                          [axes[line, 3], axes[line, 1]], 
                          [Xsulci_r[:,mode], Ycogn_r[:,mode]]):
        print(f'======== LV {LV}')
        x = np.array(x)[~pd.isnull(groups)]
        y = np.array(df_gen.Age)[~pd.isnull(groups)]
        gr = np.array(groups)[~pd.isnull(groups)]
        grs = list(set(gr))
        st, pval_age = stats.ttest_ind(y[gr == grs[0]],y[gr == grs[1]])
        print(f'Test age differences p-value={pval_age:.2e}')
        for i in set(gr):
            print(f'Mean age {dict_label[i]} {y[gr==i].mean():.2f}')
            path = ax.scatter(y[gr == i], x[gr == i],
                              label=dict_label[i], marker="o",
                              c=col[i], s=s) #75
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
        df = df_gen[covar + [between]][~pd.isnull(groups)]
        df['dv'] = x
        if len(set(gr)) > 2:
            # test mean difference
            for g, th in zip([[1, 2], [2, 3], [1, 3]], [0, 5, 15]):
                anc = ancova(data=df[df_gen.Group.isin(g)], dv='dv', covar=covar, between=between)
                if anc.loc[0, 'p-unc'] < 0.05:
                    star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
                    axh.plot([g[0] + 0.25, g[0] + 0.25, g[1] - 0.25, g[1] - 0.75],
                              [th, th, th, th], linewidth=2, color='k')
                    axh.text(sum(g)/2, th+1, star, ha='center', va='bottom', color='k')
            # test slope difference
            for grs in [[1, 2], [2, 3], [1, 3]]:
                tstat, pval = test_slopes(x[gr==grs[0]], y[gr==grs[0]], 
                                          x[gr==grs[1]], y[gr==grs[1]])
                if pval < 0.05:
                    print(f'SLOPE SIGN DIFF {dict_label[grs[0]]} vs. {dict_label[grs[1]]} pval={pval:.2e}')
                else:
                    print(f'Not slope sign diff {dict_label[grs[0]]} vs. {dict_label[grs[1]]} pval={pval:.2e}')
        else:
            grs = list(set(gr))
            # test mean difference
            anc = ancova(data=df, dv='dv', covar=covar, between=between)
            pvals.append(anc.loc[0, 'p-unc'])
            print(f'{LV} {between} {dict_label[grs[0]]} {np.mean(x[gr==grs[0]]):.2f} '+
                  f'{dict_label[grs[1]]} {np.mean(x[gr==grs[1]]):.2f} ' +
                  f'p-val={anc.loc[0, "p-unc"]:.2e} (covar={covar})')
            if anc.loc[0, 'p-unc'] < 0.05:
                star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
                axh.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
                axh.text(1.5, 1, star, ha='center', va='bottom', color='k')
            # test slope difference
            tstat, pval = test_slopes(x[gr==grs[0]], y[gr==grs[0]], 
                                      x[gr==grs[1]], y[gr==grs[1]])
            if pval < 0.05:
                print(f'SLOPE SIGN DIFF pval={pval:.2e}')
            else:
                print(f'Not slope sign diff pval={pval:.2e}')
        LV = 'COGN'
    if len(pvals) != 0:
        print()
        print(f'{between} (SW, p-value={pvals[0]:.2e}; cognition, p-value={pvals[1]:.2e})')
    return axes

# https://www.real-statistics.com/regression/hypothesis-testing-significance-regression-line-slope/comparing-slopes-two-independent-samples/
# https://www.py4u.net/discuss/192765
def test_slopes(X1, Y1, X2, Y2):
    # fit models
    model1 = sm.OLS(Y1,np.vstack([np.ones(len(X1)),X1]).T)
    results1 = model1.fit()
    b1 = results1.params[1]
    syx1 = np.sqrt(results1.mse_resid)
    sx1 = np.std(X1)
    sb1 = syx1/(sx1*np.sqrt(len(X1)-1))
    model2 = sm.OLS(Y2,np.vstack([np.ones(len(X2)),X2]).T)
    results2 = model2.fit()
    b2 = results2.params[1]
    syx2 = np.sqrt(results1.mse_resid)
    sx2 = np.std(X2)
    sb2 = syx2/(sx2*np.sqrt(len(X2)-1))
    # stat test
    tstat = (b1-b2)/np.sqrt(sb1**2+sb2**2)
    df = len(X1)+len(X2)-4
    pval = stats.t.sf(tstat, df)*2
    return tstat, pval

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

pipeline.fit(Xsulci, Ycogn)
Xsulci_r, Ycogn_r = pipeline.transform(Xsulci, Ycogn)
linreg = stats.linregress(df_gen.loc[df_gen.Group == 1, 'Age'], Xsulci_r[:,0])
pval_sulci = linreg.pvalue
linreg = stats.linregress(df_gen.loc[df_gen.Group == 1, 'Age'], Ycogn_r[:,0])
pval_cogn = linreg.pvalue
print(f'AGE CORRELATION (SW, p-value={pval_sulci:.2e}; cognition, p-value={pval_cogn:.2e})')
print()
Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)

plot_age_effect(Xsulci_r, Ycogn_r, df_gen)

# amyloid and apoe after regressing out
pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

pipeline.fit(Xsulci, Ycogn)
Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)

df_hc = df_gen[df_gen.Group == 1]
t = stats.ttest_ind(Xsulci_r[df_gen.Group == 1][df_hc.Amyloid == 'Positive'],
                    Xsulci_r[df_gen.Group == 1][df_hc.Amyloid == 'Negative'])
print(f'AMYLOID SW pval={t.pvalue[0]:.2e}')
t = stats.ttest_ind(Ycogn_r[df_gen.Group == 1][df_hc.Amyloid == 'Positive'],
                    Ycogn_r[df_gen.Group == 1][df_hc.Amyloid == 'Negative'])
print(f'AMYLOID COGN pval={t.pvalue[0]:.2e}')
covar = ['PRS_noAPOE']
df_twin = df_gen[df_gen.Twin == 'MZ']
twin_list = []
for idx in df_twin.index:
    if not df_twin.loc[idx, 'Twin_ID'] in twin_list:
        twin_list.append(idx)
rm_list = [idx for idx in df_twin.index if idx not in twin_list]
select = np.array([True if row['Group'] == 1 and idx not in rm_list else False for idx, row in df_gen.iterrows()])
between = 'APOEe4'
df = df_gen.loc[select, covar + [between]]
df['dv'] = Xsulci_r[select]
anc = ancova(data=df, dv='dv', covar=covar, between=between)
print(f'APOE SW pval={anc.loc[0, "p-unc"]:.2e}')
df['dv'] = Ycogn_r[select]
anc = ancova(data=df, dv='dv', covar=covar, between=between)
print(f'APOE COGN pval={anc.loc[0, "p-unc"]:.2e}')

#######################
# LOADINGS COMPARISON #
#######################

remove_missing = True
downsampling = True
mean = True
cogn_startswith = None

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

x_loadings, y_loadings, lv_sulci, lv_cogn = [], [], [], []
boostx, boosty = [], []
for database, confound, train_hc_only in zip(['pisa', 'aibl', 'adni', 'pisa'],
                                             [False, False, False, True],
                                             [True, True, True, False]):
    data_path = f'/home/leonie/Documents/data/{database}'
    df_sulci = pd.read_csv(f'{data_path}/sMRI/sulci_morphometry_measurements/per_measurement/{measure}.csv', index_col=0)
    df_cogn = pd.read_csv(f'{data_path}/cognitive_scores/neuropsych.csv', index_col=0)
    df_gen = pd.read_csv(f'{data_path}/cognitive_scores/demographics.csv', index_col=0)

    df_sulci, df_cogn, df_gen, Xsulci, Ycogn = preprocessing(
        df_sulci, df_cogn, df_gen, regress_out_confounds=confound,
        train_hc_only=train_hc_only, mean=True, drop_rate=0.5)

    Xsulci_r, Ycogn_r = pipeline.fit_transform(Xsulci, Ycogn)
    x_loadings.append([stats.pearsonr(Xsulci_r[:, 0], Xsulci[col].fillna(Xsulci[col].mean()))[0] for col in df_sulci.columns])
    y_loadings.append([stats.pearsonr(Ycogn_r[:, 0], Ycogn[col].fillna(Ycogn[col].mean()))[0] for col in df_cogn.columns])
    Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)
    lv_sulci.append(Xsulci_r)
    lv_cogn.append(Ycogn_r)
    xw, yw, xl, yl = pls.bootstrapping(Xsulci, Ycogn, pipeline)
    boostx.append(xl)
    boosty.append(yl)


# snapshot
comp = [[3, 0], [1, 0], [2, 0]]
# comp = [[1, 2], [1, 3], [2, 3]] # supplementary material
for i, j in comp:
    scores = np.array(x_loadings)
    sc = np.array(boostx[i]).mean(axis=0)[:, 0] - np.array(boostx[j]).mean(axis=0)[:, 0]
    dict_sulcus = {s+'_left': x for s,x in zip(Xsulci.columns, sc) if s!=0}
    for s in Xsulci.columns:
        dict_sulcus[s+'_left'] = dict_sulcus[s+'_left']
    dict_reg = {0 : [0.5, -0.5, -0.5, 0.5], 1 : [0.5, 0.5, 0.5, 0.5]}
    for side in ['left']:
        for reg in [0, 1]:
            view_sulcus_scores(
                dict_sulcus,
                side=side,
                reg_q=dict_reg[reg],
                minVal=-0.07, maxVal=0.07,
                snapshot=f'/tmp/comparision_{i}{j}_{side}{reg}.png')

corr = pd.DataFrame(np.corrcoef([np.array(boostx[i]).mean(axis=0)[:, 0] for i in range(4)]),
                    columns=['PISA HC', 'AIBL HC', 'ADNI HC', 'PISA ALL'],
                    index=['PISA HC', 'AIBL HC', 'ADNI HC', 'PISA ALL'])
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.set(font_scale=1.2)
ax = sns.heatmap(corr, square=True, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

############
# AGE PLOT #
############

df = pd.DataFrame()
group_dict = {1: 'HC', 2: 'CC', 3: 'CC'}
for database, db in zip(['pisa', 'aibl', 'adni'],
                        ['PISA', 'AIBL', 'ADNI']):
    data_path = f'/home/leonie/Documents/data/{database}'
    df_sulci = pd.read_csv(f'{data_path}/sMRI/sulci_morphometry_measurements/per_measurement/{measure}.csv', index_col=0)
    df_cogn = pd.read_csv(f'{data_path}/cognitive_scores/neuropsych.csv', index_col=0)
    df_gen = pd.read_csv(f'{data_path}/cognitive_scores/demographics.csv', index_col=0)

    df_sulci, df_cogn, df_gen, Xsulci, Ycogn = preprocessing(
        df_sulci, df_cogn, df_gen, regress_out_confounds=False,
        train_hc_only=False, mean=True, drop_rate=0.5)

    df_gen['Group'] = [f'{db} {group_dict[g]}' for g in df_gen.Group]
    df = pd.concat([df, df_gen[['Age', 'Group']]], axis=0)
df['age'] = df.Age

# figure
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pal_blue = sns.cubehelix_palette(10, rot=-.25, light=.7)
pal_red = sns.cubehelix_palette(10, light=.7)
pal = [pal_blue[i] if i % 2 == 0 else pal_red[i] for i in range(10)]
g = sns.FacetGrid(df, row="Group", hue="Group", aspect=15, height=.5, palette=pal)
g.map(sns.kdeplot, "age",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "age", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)
def label(x, color, label):
    ax = plt.gca()
    ax.text(0.89, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
g.map(label, "age")
g.fig.subplots_adjust(hspace=-.25)
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

##################
# MISSING VALUES #
##################

dict_cogn = {}
for c in df_cogn.columns:
    if not c.endswith('*(-1)'):
        dict_cogn[c] = c[2:]
    else:
        dict_cogn[c] = c[2:-6]
full_labels = df_cogn.columns
means = [x/len(df_cogn) for x in df_cogn.isna().sum()]
full_labels = df_cogn.columns
full_labels = [l for m, l in sorted(zip(means, full_labels))]
labels = [dict_cogn[s] if s in dict_cogn.keys() else s for s in full_labels] 
probas = [m for m in sorted(means)]
colors = []
for l in full_labels:
    if l.startswith('M.'):
        colors.append('lightcoral')
    elif l.startswith('L.'):
        colors.append('lightgreen')
    elif l.startswith('E.'):
        colors.append('lightblue')
    else:
        colors.append('plum')
plt.rcParams.update({'font.size': 40})
if database == 'adni':
    fig, ax = plt.subplots(figsize=(30, 40))
else:
    fig, ax = plt.subplots(figsize=(25, 30))
ax.barh(range(len(labels)), probas, 0.9, color=colors, align='center')#, edgecolor='k', linewidth=1)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', label='Executive Functions'),
                   Patch(facecolor='lightcoral', label='Memory'),
                   Patch(facecolor='lightgreen', label='Language'),
                   Patch(facecolor='plum', label='Other')]
ax.legend(handles=legend_elements, loc='center right')
ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
ax.set_xticklabels(['0%', '5%', '10%', '15%', '20%', '25%'])
ax.set_xlim(0, max(means))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
fig.tight_layout()

#AMYLOID
Counter(df_gen.loc[df_gen.Group == 1, 'Amyloid'])

# APOE
c = Counter(df_gen.loc[df_gen.Group == 1,'APOEe4'])
print(f'HC with missing APOE: {100*(1-(c[4]+c[0])/len(df_gen[df_gen.Group == 1])):.0f} %')
print(f'HC with e4 allele: {100*c[4]/(c[4]+c[0]):.0f} %')

##################
# POWER ANALYSIS #
##################

database = 'pisa'
regress_out_confounds = False
train_hc_only = True
measure='opening'

database = 'aibl'
regress_out_confounds = False
train_hc_only = True
measure='opening'

database = 'adni'
regress_out_confounds = False
train_hc_only = True
measure='opening'

database = 'pisa'
regress_out_confounds = True
train_hc_only = False
measure='opening'


data_path = f'/home/leonie/Documents/data/{database}'
df_sulci = pd.read_csv(f'{data_path}/sMRI/sulci_morphometry_measurements/per_measurement/{measure}.csv', index_col=0)
df_cogn = pd.read_csv(f'{data_path}/cognitive_scores/neuropsych.csv', index_col=0)
df_gen = pd.read_csv(f'{data_path}/cognitive_scores/demographics.csv', index_col=0)

df_sulci, df_cogn, df_gen, Xsulci, Ycogn = preprocessing(
    df_sulci, df_cogn, df_gen, regress_out_confounds=regress_out_confounds,
    train_hc_only=train_hc_only, mean=True, drop_rate=0.5)

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

pipeline.fit(Xsulci, Ycogn)
Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)

from pingouin import power_anova
from sklearn.linear_model import LinearRegression
import pingouin as pg

for X, LV in zip([Xsulci_r, Ycogn_r], ['SW', 'COGN']):
    print()
    print(f'===== {LV}')
    # amyloid
    df = df_gen[df_gen.Group == 1].copy()
    y = X[df_gen.Group == 1]
    if not regress_out_confounds:
        df_cov = df[['Age', 'Sex']].copy()
        reg = LinearRegression().fit(df_cov, y)
        df.loc[:,LV] = y - reg.predict(df_cov)
    else:
        df.loc[:,LV] = y
    if np.mean(df.loc[df.Amyloid == 'Positive', LV]) > np.mean(df.loc[df.Amyloid == 'Negative', LV]):
        print('POS >> NEG')
    else:
        print('NEG >> POS')
    aov = pg.anova(dv=LV, between='Amyloid', data=df, detailed=True)
    p = power_anova(eta=aov.loc[0, 'np2'], k=2, n=len(df.dropna(subset=['Amyloid'])))
    n = power_anova(eta=aov.loc[0, 'np2'], k=2, power=0.80)
    print(f'AMYLOID sample size: {n:.2f}; power: {p:.2f}')

    # apoe
    if not regress_out_confounds:
        covar = ['Age', 'Sex']
    else:
        covar = []
    if 'Twin' in df_gen.columns:
        df_twin = df_gen[df_gen.Twin == 'MZ']
        twin_list = []
        for idx in df_twin.index:
            if not df_twin.loc[idx, 'Twin_ID'] in twin_list:
                twin_list.append(idx)
        rm_list = [idx for idx in df_twin.index if idx not in twin_list]
        select = np.array([True if row['Group'] == 1 and idx not in rm_list else False for idx, row in df_gen.iterrows()])
    else:
        select = np.array(df_gen.Group == 1)
    df = df_gen[select].copy()
    y = X[select]
    if 'PRS_noAPOE' in df_gen.columns:
        covar = covar + ['PRS_noAPOE']
        y = y[~df.PRS_noAPOE.isna()]
        df.dropna(subset=['PRS_noAPOE'], inplace=True)
    if len(covar) != 0:
        df_cov = df[covar].copy()
        reg = LinearRegression().fit(df_cov, y)
        df.loc[:,LV] = y - reg.predict(df_cov)
    else:
        df.loc[:,LV] = y
    if np.mean(df.loc[df.APOEe4 == 4, LV]) > np.mean(df.loc[df.APOEe4 == 0, LV]):
        print('E4 >> OTHER')
    else:
        print('OTHER >> E4')
    aov = pg.anova(dv=LV, between='APOEe4', data=df, detailed=True)
    p = power_anova(eta=aov.loc[0, 'np2'], k=2, n=len(df.dropna(subset=['APOEe4'])))
    n = power_anova(eta=aov.loc[0, 'np2'], k=2, power=0.80)
    print(f'APOEe4 sample size: {n:.2f}; power: {p:.2f}')

