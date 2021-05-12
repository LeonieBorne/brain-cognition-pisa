import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu, pearsonr, linregress
from scipy import stats 

from sklearn.utils import resample
from sklearn.cross_decomposition import PLSCanonical, CCA
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

import uon.preprocessing as prep
import uon.pls as pls
from uon.plot import plot_values, plot_groups
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

import time
from tqdm import tqdm
from pingouin import ancova

plt.style.use('default')
# plt.rcParams.update({'font.size': 18})



#####################
# F1 - PLS LOADINGS #
#####################
database = 'aibl'
confound = False
train_hc_only = True
remove_missing = True
downsampling = True
mean = True
cogn_startswith = None

df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                           cogn_startswith=cogn_startswith)
if database == 'aibl':
    df_gen['Group'] = [1 if c == 'HC' else c for c in df_gen.classification]
    df_gen['Group'] = [2 if c in ['MCI', 'MCIX'] else c for c in df_gen.Group]
    df_gen['Group'] = [3 if c in ['AD', 'DNOS', 'FTD', 'PD', 'VD'] else c for c in df_gen.Group]
    df_sulci = df_sulci.loc[[idx for idx in df_sulci.index if idx.endswith('_bl')]]
    df_cogn = df_cogn.loc[[idx for idx in df_cogn.index if idx.endswith('_bl')]]
    df_gen = df_gen.loc[[idx for idx in df_gen.index if idx.endswith('_bl')]]
if remove_missing:
    df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
if downsampling:
    df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
    df_sulci, df_cogn, df_gen, confound, train_hc_only)

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

x_weights, y_weights, x_loadings, y_loadings = pls.bootstrapping(Xsulci, Ycogn, pipeline)

dict_cogn = {'M.MMSE': 'MMSE',
 'E.Total fwd + bk': 'Digital Span Test',
 'E.Raw score (digit symbol coding)': 'Digital Symbol Coding Score',
 'M.Recall RAW (LM1)': 'Logical Memory Test 1',
 'M.Recall RAW (LMII)': 'Logical Memory Test 2',
 'M.List A 1-5 RAW': 'CVLT: 1-5 raw score (List A)',
 'M.List A 1-5 T score': 'CVLT: 1-5 T score (List A)',
 'M.List A T6 Retention (RAW)': 'CVLT: immediate free recall (List A)',
 'M.List A Delayed Recall (RAW)': 'CVLT: delayed free recall (List A)',
 'M.List A Recognition (RAW)': 'CVLT: recognition (List A)',
 'M.m.List A False Positives (RAW)': 'CVLT: false positives (List A)',
 'M.Total recog discrim d (RAW)': 'CVLT: total recognition discrim',
 'M.RCFT Copy (RAW)': 'RCFT: copy score',
 'M.m.RCFT Copy time (RAW)': 'RCFT: copy time score',
 'M.RCFT 3 min delay (RAW)': 'RCFT: 3 min delay score',
 'M.RCFT 30 min delay (RAW)': 'RCFT: 30 min delay score',
 'M.RCFT Recog (RAW)': 'RCFT: recognition score',
 'E.FAS (total)': 'FAS: total correct', # D-KEFS verbal fluency
 'L.Animals + Names (RAW)': 'Animal: total correct', #D-KEFS category fluency total correct for animals+boys’ names
 'L.Fruit/furniture Total (RAW)': 'Fruit/furniture: total correct', #D-KEFS verbal fluency
 'L.Fruit/furniture Switching (RAW)': 'Fruit/furniture: switching', #D-KEFS verbal fluency
 'L.BNT - No Cue (Australian RAW)': 'L.BNT - No Cue (Australian RAW)', 
 'L.m.BNT - Stimulus Cued Score (Australian)': 'BNT: stimulus cued score',
 'L.m.BNT - Phonemic Cued Score (Australian)': 'BNT: phonemic cued score',
 'M.Clock Score': 'Clock score',
 'L.Score': 'Wechsler Test of Adult Reading (WTAR)',
 'E.m.Dots time (RAW)': 'Stroop: dot colour naming reaction time',
 'E.m.Dots errs': 'E.m.Dots errs',
 'E.m.Words time (RAW)': 'Stroop: word colour naming reaction time',
 'E.m.Words errs': 'E.m.Words errs',
 'E.m.Colours time (RAW)': 'Stroop: colour naming reaction time',
 'E.m.Colours errs': 'E.m.Colours errs',
 'E.m.C/D Stroop (RAW)': 'E.m.C/D Stroop (RAW)',
 'O.m.HADS (D)': 'HADS: depression',
 'O.m.HADS (A)': 'HADS: anxiety'}

# cognitive weights
full_labels = Ycogn.columns
means = [m if up*down > 0 else 0 for m, up, down in zip(
    np.mean(y_loadings, axis=0)[:, 0],
    np.percentile(y_loadings, 97.5, axis=0)[:, 0],
    np.percentile(y_loadings, 2.5, axis=0)[:, 0])]
full_labels = Ycogn.columns
labels = [dict_cogn[s] if s in dict_cogn.keys() else s for s in full_labels] 
labels = [l for m, l in sorted(zip(means, labels)) if m != 0]
full_labels = [l for m, l in sorted(zip(means, full_labels)) if m != 0]
probas = [ m for m in sorted(means) if m != 0]
colors = []
for l in full_labels:
    if l.startswith('M.'):
        colors.append('lightcoral')
    elif l.startswith('L.'):
        colors.append('lightgreen')
    elif l.startswith('E.'):
        colors.append('lightblue')
    elif l.startswith('S.') or l.startswith('O.'):
        colors.append('plum')
    else:
        colors.append('moccasin')
        
plt.rcParams.update({'font.size': 40})
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_yticklabels([]) # Hide the left y-axis tick-labels
ax.set_yticks([]) # Hide the left y-axis ticks
ax1 = ax.twinx() # Create a twin x-axis
ax1.barh(range(len(labels)), probas, 0.9, color=colors, align='center')#, edgecolor='k', linewidth=1)
ax1.set_yticks(range(len(labels)))
ax1.set_yticklabels(labels)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', #edgecolor='k',
                         label='Executive Functions'),
                   Patch(facecolor='lightcoral', #edgecolor='k',
                         label='Memory'),
                   Patch(facecolor='lightgreen', #edgecolor='k',
                         label='Language'),
                   Patch(facecolor='plum', #edgecolor='k',
                         label='Other')]
ax1.legend(handles=legend_elements, loc='best')
ax1.set_xticks([0, -0.1, -0.2, -0.3])
ax1.set_ylim(-0.5, len(labels)-0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# SW weights
from uon.snapshots import view_sulcus_scores
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
            snapshot=f'/tmp/{side}{reg}_bootstrap.png')

sorted([[v,k] for k, v in dict_sulcus.items()])


###################
# F2 - AGE EFFECT #
###################

database = 'aibl'
confound = False
train_hc_only = True
remove_missing = True
downsampling = True
mean = True
cogn_startswith = None

df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                           cogn_startswith=cogn_startswith)
if database == 'aibl':
    df_gen['Group'] = [1 if c == 'HC' else c for c in df_gen.classification]
    df_gen['Group'] = [2 if c in ['MCI', 'MCIX'] else c for c in df_gen.Group]
    df_gen['Group'] = [3 if c in ['AD', 'DNOS', 'FTD', 'PD', 'VD'] else c for c in df_gen.Group]
    df_sulci = df_sulci.loc[[idx for idx in df_sulci.index if idx.endswith('_bl')]]
    df_cogn = df_cogn.loc[[idx for idx in df_cogn.index if idx.endswith('_bl')]]
    df_gen = df_gen.loc[[idx for idx in df_gen.index if idx.endswith('_bl')]]
if remove_missing:
    df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
if downsampling:
    df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
    df_sulci, df_cogn, df_gen, confound, train_hc_only)

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

pipeline.fit(Xsulci, Ycogn)
Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)

data = np.hstack([Xsulci_r, Ycogn_r])
df_gen['APOEe4'] = [4 if a in ['E3/E4', 'E4/E2', 'E4/E3', 'E4/E4'] else a for a in df_gen.ApoE]
df_gen['APOEe4'] = [0 if a in ['E2/E2', 'E3/E2', 'E3/E3'] else a for a in df_gen.APOEe4]
df_gen['Amyloid'] = ['Uncertain' if (pd.isnull(a) or a == 'Na') else a for a in df_gen['Amyloid Status']]
df_gen['Amyloid'] = ['Positive' if a.startswith('P') else a for a in df_gen['Amyloid']]
df_gen['Amyloid'] = ['Negative' if a.startswith('N') else a for a in df_gen['Amyloid']]
df_gen['Amyloid'] = [np.nan if (pd.isnull(a) or a == 'Uncertain') else a for a in df_gen['Amyloid']]


plt.style.use('default')
plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(4, 4, figsize=(30, 20), sharey='row',
                         gridspec_kw={'width_ratios': [4, 2, 4, 2]})
# groups
groups=df_gen.Group
col={1:'steelblue', 3:'tab:orange', 2:'red'}
dict_label={1:'HC', 2:'MCI', 3:'AD'}
y=df_gen.Age
for ax, axh, x in zip([axes[0, 0], axes[0, 2]], 
                      [axes[0, 1], axes[0, 3]], 
                      [Ycogn_r[:,0], Xsulci_r[:,0]]):
    for i in set(groups):
        path = ax.scatter(y[groups == i], x[groups == i],
                          label=dict_label[i], marker="o",
                          c=col[i], s=75)
        Xg, Yg = x[groups == i], y[groups == i]
        linreg = stats.linregress(Yg[~np.isnan(Yg)],
                                  Xg[~np.isnan(Yg)])
        if linreg.pvalue < 0.05:
            ax.plot(y, linreg.intercept + linreg.slope*y, 
                    color=path.get_facecolors()[0].tolist(),
                    linewidth=4)
        # axh.hist(x[groups == i], alpha=0.5, bins=5,
        #          orientation='horizontal', color=col[i])
    data = [list(x[groups == g]) for g in sorted(list(set(groups)))]
    labels = [dict_label[g] for g in sorted(list(set(groups)))]
    colors = [col[g] for g in sorted(list(set(groups)))]
    bplot = axh.boxplot(data, notch=True, vert=True,
                        patch_artist=True, whis=100, labels=labels)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age', 'Group']]
    df['dv'] = x
    for gr, th in zip([[1, 2], [2, 3], [1, 3]], [0, 5, 15]):
        anc = ancova(data=df[df_gen.Group.isin(gr)], dv='dv', covar=['Age', 'Sex'], between='Group')
        if anc.loc[0, 'p-unc'] < 0.01:
            star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
            axh.plot([gr[0] + 0.25, gr[0] + 0.25, gr[1] - 0.25, gr[1] - 0.75],
                     [th, th, th, th], linewidth=2, color='k')
            axh.text(sum(gr)/2, th+1, star, ha='center', va='bottom', color='k')
axes[0, 0].legend(loc='lower left', ncol=3)
axes[0, 2].set_title('Sulcal Width')
axes[0, 0].set_title('Cognitive Scores')
# sex
groups=df_gen.Sex[df_gen.Group == 1]
col={0:'salmon', 1:'lightseagreen'}
dict_label={0:'Female', 1:'Male'}
y=df_gen.Age[df_gen.Group == 1]
for ax, axh, x in zip([axes[1, 0], axes[1, 2]], 
                      [axes[1, 1], axes[1, 3]], 
                 [Ycogn_r[df_gen.Group == 1,0], Xsulci_r[df_gen.Group == 1,0]]):
    for i in set(groups):
        path = ax.scatter(y[groups == i], x[groups == i],
                          label=dict_label[i], marker="o",
                          c=col[i], s=75)
        Xg, Yg = x[groups == i], y[groups == i]
        linreg = stats.linregress(Yg[~np.isnan(Yg)],
                                  Xg[~np.isnan(Yg)])
        if linreg.pvalue < 0.05:
            ax.plot(y, linreg.intercept + linreg.slope*y, 
                    color=path.get_facecolors()[0].tolist(),
                    linewidth=4)
        # axh.hist(x[groups == i], alpha=0.5, bins=5,
        #          orientation='horizontal', color=col[i])
        # df = df_gen[['Sex', 'Age']][df_gen.Group == 1]
        # df['dv'] = x
        # anc = ancova(data=df, dv='dv', covar=['Age'], between='Sex')
        # if anc.loc[0, 'p-unc'] < 0.05:
        #     star = '*\n*' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        #     axh.plot([50, 50], [np.mean(x[groups == i]) for i in set(groups)],
        #              linewidth=2, color='k')
        #     axh.text(50+10, 0, star, ha='center', va='bottom', color='k')
    data = [list(x[groups == g]) for g in sorted(list(set(groups)))]
    labels = [dict_label[g] for g in sorted(list(set(groups)))]
    colors = [col[g] for g in sorted(list(set(groups)))]
    bplot = axh.boxplot(data, notch=True, vert=True, widths=0.3,
                       patch_artist=True, whis=100, labels=labels)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age']][df_gen.Group == 1]
    df['dv'] = x
    anc = ancova(data=df, dv='dv', covar=['Age'], between='Sex')
    if anc.loc[0, 'p-unc'] < 0.01:
        star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        axh.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
        axh.text(1.5, 1, star, ha='center', va='bottom', color='k')
axes[1, 0].legend(loc='upper left')
# amyloid
groups=df_gen.Amyloid[df_gen.Group == 1]
col={'Negative':'sandybrown', 'Positive':'darkseagreen'}
dict_label={'Negative':'Aβ negative', 'Positive':'Aβ positive'}
y=df_gen.Age[df_gen.Group == 1]
for ax, axh, x in zip([axes[2, 0], axes[2, 2]], 
                      [axes[2, 1], axes[2, 3]], 
                 [Ycogn_r[df_gen.Group == 1,0], Xsulci_r[df_gen.Group == 1,0]]):
    x = np.array(x)[~pd.isnull(groups)]
    y = np.array(df_gen.Age[df_gen.Group == 1])[~pd.isnull(groups)]
    gr = np.array(groups)[~pd.isnull(groups)]
    for i in ['Negative', 'Positive']:
        path = ax.scatter(y[gr == i], x[gr == i],
                          label=dict_label[i], marker="o",
                          c=col[i], s=75)
        Xg, Yg = x[gr == i], y[gr == i]
        linreg = stats.linregress(Yg[~np.isnan(Yg)],
                                  Xg[~np.isnan(Yg)])
        if linreg.pvalue < 0.01:
            ax.plot(y, linreg.intercept + linreg.slope*y, 
                    color=path.get_facecolors()[0].tolist(),
                    linewidth=4)
        # axh.hist(x[groups == i], alpha=0.5, bins=5,
        #          orientation='horizontal', color=col[i])
    data = [list(x[gr == g]) for g in sorted(list(set(gr)))]
    labels = [dict_label[g] for g in sorted(list(set(gr)))]
    colors = [col[g] for g in sorted(list(set(gr)))]
    bplot = axh.boxplot(data, notch=True, vert=True, widths=0.3,
                       patch_artist=True, whis=100, labels=labels)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age', 'Amyloid']][df_gen.Group == 1][~pd.isnull(groups)]
    df['dv'] = x
    anc = ancova(data=df, dv='dv', covar=['Age', 'Sex'], between='Amyloid')
    if anc.loc[0, 'p-unc'] < 0.05:
        star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        axh.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
        axh.text(1.5, 1, star, ha='center', va='bottom', color='k')
axes[2, 0].legend(loc='upper left')
# apoe
groups=df_gen.APOEe4[df_gen.Group == 1]
col={0:'orchid', 4:'lightskyblue'}
dict_label={0:'no APOE ε4', 4:'APOE ε4'}
y=df_gen.Age[df_gen.Group == 1]
for ax, axh, x in zip([axes[3, 0], axes[3, 2]], 
                      [axes[3, 1], axes[3, 3]], 
                 [Ycogn_r[df_gen.Group == 1,0], Xsulci_r[df_gen.Group == 1,0]]):
    x = np.array(x)[~pd.isnull(groups)]
    y = np.array(df_gen.Age[df_gen.Group == 1])[~pd.isnull(groups)]
    gr = np.array(groups)[~pd.isnull(groups)]
    for i in set(gr):
        path = ax.scatter(y[gr == i], x[gr == i],
                          label=dict_label[i], marker="o",
                          c=col[i], s=75)
        Xg, Yg = x[gr == i], y[gr == i]
        linreg = stats.linregress(Yg[~np.isnan(Yg)],
                                  Xg[~np.isnan(Yg)])
        if linreg.pvalue < 0.05:
            ax.plot(y, linreg.intercept + linreg.slope*y, 
                    color=path.get_facecolors()[0].tolist(),
                    linewidth=4)
        # axh.hist(x[gr == i], alpha=0.5, bins=5,
        #          orientation='horizontal', color=col[i])
    data = [list(x[gr == g]) for g in sorted(list(set(gr)))]
    labels = [dict_label[g] for g in sorted(list(set(gr)))]
    colors = [col[g] for g in sorted(list(set(gr)))]
    bplot = axh.boxplot(data, notch=True, vert=True, widths=0.3,
                       patch_artist=True, whis=100, labels=labels)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age', 'APOEe4']][df_gen.Group == 1][~pd.isnull(groups)]
    df['dv'] = x
    anc = ancova(data=df, dv='dv', covar=['Age', 'Sex'], between='APOEe4')
    if anc.loc[0, 'p-unc'] < 0.01:
        star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        axh.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
        axh.text(1.5, 1, star, ha='center', va='bottom', color='k')
axes[3, 0].legend(loc='upper left')
axes[3, 0].set_xlabel("age")
axes[3, 2].set_xlabel("age")
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
# plt.xlabel("age")
plt.ylabel("latent variable")
plt.tight_layout()


#####################
# FIG 3 RISK FACTOR #
#####################


from pingouin import ancova
def test_groups()
df = df_gen[['Age', 'Sex', 'Group', 'Amyloid', 'APOEe4', 'Education', 
             'risk_group', 'APOEe2', 'APOEv3', 'PRS_noAPOE']]
df['SW'] = np.array(data[:, 0])
df['COGN'] = np.array(data[:, 1])
df['SW+COGN'] = np.array(data[:, 2])
ancova(data=df[df.Group == 1], dv='SW+COGN', covar='Age', between='Amyloid')
ancova(data=df[df.Group == 1], dv='SW+COGN', covar='Age', between='APOEe4.Sex')
ancova(data=df[df.Group == 1], dv='SW+COGN', covar=['Age', 'Sex'], between='Sex')
ancova(data=df, dv='SW+COGN', covar=['Age', 'Sex'], between='Group')


plt.style.use('default')
plt.rcParams.update({'font.size': 25})
fig, axes = plt.subplots(4, 2, figsize=(20, 20), sharey='row')
# groups
groups=df_gen.Group
col={1:'steelblue', 2:'tab:orange', 3:'red'}
dict_label={1:'HC', 2:'MCI', 3:'AD'}
for ax, x in zip([axes[0, 0], axes[0, 1]], [Xsulci_r[:,0], Ycogn_r[:,0]]):
    data = [list(x[groups == g]) for g in sorted(list(set(groups)))]
    labels = [dict_label[g] for g in sorted(list(set(groups)))]
    colors = [col[g] for g in sorted(list(set(groups)))]
    bplot = ax.boxplot(data, notch=True, vert=True,
                       patch_artist=True, labels=labels, whis=100)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
axes[0, 1].legend(loc='upper right')
axes[0, 0].set_title('Sulcal Width')
axes[0, 1].set_title('Cognitive Scores')
# sex
groups=df_gen.Sex[df_gen.Group == 1]
col={0:'salmon', 1:'lightseagreen'}
dict_label={0:'Female', 1:'Male'}
y=df_gen.Age[df_gen.Group == 1]
for ax, x in zip([axes[1, 0], axes[1, 1]], 
                 [Xsulci_r[df_gen.Group == 1,0], Ycogn_r[df_gen.Group == 1,0]]):
    data = [list(x[groups == g]) for g in sorted(list(set(groups)))]
    labels = [dict_label[g] for g in sorted(list(set(groups)))]
    colors = [col[g] for g in sorted(list(set(groups)))]
    bplot = ax.boxplot(data, notch=True, vert=True,
                       patch_artist=True, labels=labels, whis=100)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age']][df_gen.Group == 1]
    df['dv'] = x
    anc = ancova(data=df, dv='dv', covar=['Age'], between='Sex')
    if anc.loc[0, 'p-unc'] < 0.05:
        star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        ax.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
        ax.text(1.5, 1, star, ha='center', va='bottom', color='k')
axes[1, 1].legend(loc='upper right')
# amyloid
groups=df_gen.Amyloid[df_gen.Group == 1]
col={'Positive':'sandybrown', 'Negative':'darkseagreen'}
dict_label={'Negative':'Amyloid negative', 'Positive':'Amyloid positive'}
y=df_gen.Age[df_gen.Group == 1]
for ax, x in zip([axes[2, 0], axes[2, 1]], 
                 [Xsulci_r[df_gen.Group == 1,0], Ycogn_r[df_gen.Group == 1,0]]):
    x = np.array(x)[~pd.isnull(groups)]
    gr = np.array(groups)[~pd.isnull(groups)]
    data = [list(x[gr == g]) for g in sorted(list(set(gr)))]
    labels = [dict_label[g] for g in sorted(list(set(gr)))]
    colors = [col[g] for g in sorted(list(set(gr)))]
    bplot = ax.boxplot(data, notch=True, vert=True,
                       patch_artist=True, labels=labels, whis=100)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age', 'Amyloid']][df_gen.Group == 1][~pd.isnull(groups)]
    df['dv'] = x
    anc = ancova(data=df, dv='dv', covar=['Age', 'Sex'], between='Amyloid')
    if anc.loc[0, 'p-unc'] < 0.05:
        star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        ax.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
        ax.text(1.5, 1, star, ha='center', va='bottom', color='k')
axes[2, 1].legend(loc='upper right')
# apoe - VERIFIER LES NAN
groups=df_gen.APOEe4[df_gen.Group == 1]
col={0:'orchid', 4:'lightskyblue'}
dict_label={0:'APOE e4 non carrier', 4:'APOE e4 carrier'}
y=df_gen.Age[df_gen.Group == 1]
for ax, x in zip([axes[3, 0], axes[3, 1]], 
                 [Xsulci_r[df_gen.Group == 1,0], Ycogn_r[df_gen.Group == 1,0]]):
    data = [list(x[groups == g]) for g in sorted(list(set(groups)))]
    labels = [dict_label[g] for g in sorted(list(set(groups)))]
    colors = [col[g] for g in sorted(list(set(groups)))]
    bplot = ax.boxplot(data, notch=True, vert=True,
                       patch_artist=True, labels=labels, whis=100)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    df = df_gen[['Sex', 'Age', 'APOEe4']][df_gen.Group == 1]
    df['dv'] = x
    anc = ancova(data=df, dv='dv', covar=['Age', 'Sex'], between='APOEe4')
    if anc.loc[0, 'p-unc'] < 0.05:
        star = '**' if anc.loc[0, 'p-unc'] < 0.001 else '*'
        ax.plot([1.25,1.25, 1.75,1.75], [0, 0, 0, 0], linewidth=2, color='k')
        ax.text(1.5, 1, star, ha='center', va='bottom', color='k')
axes[3, 1].legend(loc='upper right')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel("latent variable")
plt.tight_layout()


##################################
# FIG. COMPARISON BRAIN PATTERNS #
##################################

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

x_loadings, y_loadings, lv_sulci, lv_cogn = [], [], [], []
boostx, boosty = [], []
for database, confound, train_hc_only in zip(['pisa', 'aibl', 'pisa'],
                                             [False, False, True],
                                             [True, True, False]):
    df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                               cogn_startswith=cogn_startswith)
    if database == 'aibl':
        df_gen['Group'] = [1 if c == 'HC' else c for c in df_gen.classification]
        df_gen['Group'] = [2 if c in ['MCI', 'MCIX'] else c for c in df_gen.Group]
        df_gen['Group'] = [3 if c in ['AD', 'DNOS', 'FTD', 'PD', 'VD'] else c for c in df_gen.Group]
        df_sulci = df_sulci.loc[[idx for idx in df_sulci.index if idx.endswith('_bl')]]
        df_cogn = df_cogn.loc[[idx for idx in df_cogn.index if idx.endswith('_bl')]]
        df_gen = df_gen.loc[[idx for idx in df_gen.index if idx.endswith('_bl')]]
    if remove_missing:
        df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
    if downsampling:
        df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
    df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
        df_sulci, df_cogn, df_gen, confound, train_hc_only)

    Xsulci_r, Ycogn_r = pipeline.fit_transform(Xsulci, Ycogn)
    x_loadings.append([pearsonr(Xsulci_r[:, 0], Xsulci[col].fillna(Xsulci[col].mean()))[0] for col in df_sulci.columns])
    y_loadings.append([pearsonr(Ycogn_r[:, 0], Ycogn[col].fillna(Ycogn[col].mean()))[0] for col in df_cogn.columns])
    Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)
    lv_sulci.append(Xsulci_r)
    lv_cogn.append(Ycogn_r)
    xw, yw, xl, yl = pls.bootstrapping(Xsulci, Ycogn, pipeline)
    boostx.append(xl)
    boosty.append(yl)


corr = pd.DataFrame(np.corrcoef(x_loadings), 
                    columns=['PISA HC', 'AIBL HC', 'PISA AD'],
                    index=['PISA HC', 'AIBL HC', 'PISA AD'])
import seaborn as sns
sns.set(font_scale=1.5)
ax = sns.heatmap(corr, annot=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# aibl ressemble plus a pisa apres regression out que sans regression out
plt.plot(x_loadings[0], x_loadings[1], 'o')
plt.plot(x_loadings[1], x_loadings[2], 'o')
plt.plot(x_loadings[0], x_loadings[2], 'o')














database = 'pisa'
confound = False
train_hc_only = True
remove_missing = True
downsampling = True
mean = True
cogn_startswith = None

df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                           cogn_startswith=cogn_startswith)
if database == 'aibl':
    df_gen['Group'] = [1 if c == 'HC' else c for c in df_gen.classification]
    df_gen['Group'] = [2 if c in ['MCI', 'MCIX'] else c for c in df_gen.Group]
    df_gen['Group'] = [3 if c in ['AD', 'DNOS', 'FTD', 'PD', 'VD'] else c for c in df_gen.Group]
    df_sulci = df_sulci.loc[[idx for idx in df_sulci.index if idx.endswith('_bl')]]
    df_cogn = df_cogn.loc[[idx for idx in df_cogn.index if idx.endswith('_bl')]]
    df_gen = df_gen.loc[[idx for idx in df_gen.index if idx.endswith('_bl')]]
if remove_missing:
    df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
if downsampling:
    df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
    df_sulci, df_cogn, df_gen, confound, train_hc_only)

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

pipeline.fit(Xsulci, Ycogn)
Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)
LV = np.mean(np.hstack([Xsulci_r, Ycogn_r]), axis=1)

data = np.hstack([Xsulci_r, Ycogn_r, LV.reshape(-1, 1)])

if database == 'pisa':
    df_gen['APOEe4'] = [4 if a in [24, 34, 44] else a for a in df_gen.APOEv3]
    df_gen['APOEe4'] = [a if a == 4 else 0 for a in df_gen.APOEe4]
    df_gen['APOEe2'] = [2 if a in [24, 23] else a for a in df_gen.APOEv3]
    df_gen['APOEe2'] = [a if (np.isnan(a) or a == 2) else 0 for a in df_gen.APOEe2]
    df_gen['Amyloid'] = [np.nan if (pd.isnull(a) or a == 'Uncertain') else a for a in df_gen['Amyloid Status']]
    risk_group = [1 if x=='low' else x for x in df_gen['classification of risk']]
    risk_group = [2 if x in ['high', 'high '] else x for x in risk_group]
    df_gen['risk_group'] = np.array([x if x in [1, 2] else np.nan for x in risk_group])
else:
    df_gen['APOEe4'] = [4 if a in ['E3/E4', 'E4/E2', 'E4/E3', 'E4/E4'] else a for a in df_gen.ApoE]
    df_gen['APOEe4'] = [a if (pd.isnull(a) or a == 4) else 0 for a in df_gen.APOEe4]
    df_gen['APOEe2'] = [2 if a in ['E2/E2', 'E3/E2', 'E4/E2'] else a for a in df_gen.ApoE]
    df_gen['APOEe2'] = [a if (pd.isnull(a) or a == 2) else 0 for a in df_gen.APOEe2]
    df_gen['Amyloid'] = ['Uncertain' if (pd.isnull(a) or a == 'Na') else a for a in df_gen['Amyloid Status']]
    df_gen['Amyloid'] = ['Positive' if a.startswith('P') else a for a in df_gen['Amyloid']]
    df_gen['Amyloid'] = ['Negative' if a.startswith('N') else a for a in df_gen['Amyloid']]
    df_gen['Amyloid'] = [np.nan if (pd.isnull(a) or a == 'Uncertain') else a for a in df_gen['Amyloid']]

df = df_gen[['Age', 'Sex', 'Group', 'Amyloid', 'APOEe4', 'Education', 
             'risk_group', 'APOEe2', 'APOEv3', 'PRS_noAPOE', 'Centiloid.1']]
df['SW'] = np.array(data[:, 0])
df['COGN'] = np.array(data[:, 1])
df['SW+COGN'] = np.array(data[:, 2])
df[df_gen.Group == 1].to_csv('/tmp/stat.csv')

# AGE
plot_values(data, np.array(df_gen.Age), 
            groups=df_gen.Group,
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            col={1:'r', 2:'g', 3:'b'}, 
            dict_label={1:'HC', 2:'MCI', 3:'AD'},
            invert_axes=True)

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.Amyloid[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            invert_axes=True)

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.Sex[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            col={0:'r', 1:'b'}, 
            dict_label={0:'Female', 1:'Male'},
            invert_axes=True)

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.APOEe4[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            col={0:'k', 4:'g'}, 
            dict_label={0:'e4 non carrier', 4:'e4 carrier'},
            invert_axes=True)

from pingouin import ancova
df = df_gen[['Age', 'Sex', 'Group', 'Amyloid', 'APOEe4', 'Education', 
             'risk_group', 'APOEe2', 'APOEv3', 'PRS_noAPOE']]
df['SW'] = np.array(data[:, 0])
df['COGN'] = np.array(data[:, 1])
df['SW+COGN'] = np.array(data[:, 2])
ancova(data=df[df.Group == 1], dv='SW+COGN', covar='Age', between='Amyloid')
ancova(data=df[df.Group == 1], dv='SW+COGN', covar='Age', between='APOEe4.Sex')
ancova(data=df[df.Group == 1], dv='SW+COGN', covar=['Age', 'Sex'], between='Sex')
ancova(data=df, dv='SW+COGN', covar=['Age', 'Sex'], between='Group')

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.APOEe2[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            col={0:'k', 2:'y'}, 
            dict_label={0:'e2 non carrier', 2:'e2 carrier'},
            invert_axes=True)

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.risk_group[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            col={1:'k', 2:'y'}, 
            dict_label={1:'low risk', 2:'high risk'},
            invert_axes=True)


# PRS
df_gen['APOEe4.Sex'] = [f'{int(s)}{int(a)}' for s, a in zip(df_gen.APOEe4, df_gen.Sex)]
plot_values(data[df_gen.Group == 1], np.array(df_gen.Education[df_gen.Group == 1]), 
            xlabel='LV', ylabel='Education', title=['SW', 'COGN', 'SW+COGN'],
            invert_axes=True)
plot_values(data[df_gen.Group == 1], np.array(df_gen.PRS_noAPOE[df_gen.Group == 1]), 
            xlabel='LV', ylabel='PRS no APOE', title=['SW', 'COGN', 'SW+COGN'],
            invert_axes=True)
for col in ['Amyloid', 'APOEe4', 'Sex', 'APOEe2', 'risk_group']:
    plot_groups(data[df_gen.Group == 1], 
                np.array(df_gen[col])[df_gen.Group == 1], 
                title=[f'LV SW, {col}', f'LV COGN, {col}', f'LV SW+COGN, {col}'])
plot_groups(data, 
            np.array(df_gen['Group']), 
            title=[f'LV SW, {col}', f'LV COGN, {col}', f'LV SW+COGN, {col}'])
# regress out age
from sklearn.linear_model import LinearRegression
def plot_regress_out_group(gamma, age, groups, title, hc_only=False):
    reg = LinearRegression().fit(np.array(age)[df_gen.Group == 1],
                                 np.array(gamma)[df_gen.Group == 1])
    gamma_pred = reg.predict(age)
    gamma_delta = np.array(gamma) - np.array(gamma_pred)
    X = np.array(gamma_delta)[df_gen.Group == 1] if hc_only else np.array(gamma_delta)
    G = groups[df_gen.Group == 1] if hc_only else groups
    plot_groups(X.reshape(-1, 1), G, title=title)

titles = ['REGRESS OUT AGE SW', 'REGRESS OUT AGE COGN', 'REGRESS OUT AGE SW+COGN']
for i in range(3):
    title = titles[i]
    plot_regress_out_group(data[:, i], np.array(df_gen.Age).reshape(-1, 1), df_gen.Group, [title], hc_only=False)
    plot_regress_out_group(data[:, i], np.array(df_gen.Age).reshape(-1, 1), df_gen.APOEe4, [title], hc_only=True)
    plot_regress_out_group(data[:, i], np.array(df_gen.Age).reshape(-1, 1), df_gen.Amyloid, [title], hc_only=True)

# titles = ['REGRESS OUT SEX SW', 'REGRESS OUT SEX COGN', 'REGRESS OUT SEX SW+COGN']
# for i in range(3):
#     title = titles[i]
#     # plot_regress_out_group(data[:, i], np.array(df_gen.Sex).reshape(-1, 1), df_gen.Group, [title], hc_only=False)
#     plot_regress_out_group(data[:, i], np.array(df_gen.Sex).reshape(-1, 1), df_gen.APOEe4, [title], hc_only=True)
#     # plot_regress_out_group(data[:, i], np.array(df_gen.Sex).reshape(-1, 1), df_gen.Amyloid, [title], hc_only=True)

# titles = ['REGRESS OUT SEX+AGE SW', 'REGRESS OUT SEX+AGE COGN', 'REGRESS OUT SEX+AGE SW+COGN']
# for i in range(3):
#     title = titles[i]
#     plot_regress_out_group(data[:, i], df_gen[['Sex', 'Age']], df_gen.Group, [title], hc_only=False)
#     plot_regress_out_group(data[:, i], df_gen[['Sex', 'Age']], df_gen.APOEe4, [title], hc_only=True)
#     plot_regress_out_group(data[:, i], df_gen[['Sex', 'Age']], df_gen.Amyloid, [title], hc_only=True)


# POWER ANALYSIS
from statsmodels.stats.power import TTestIndPower

effect = 0.8
alpha = 0.05
power = 0.8
analysis = TTestIndPower()
result = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)
	
# EFFECT SIZE calculate the Cohen's d between two samples
from numpy.random import randn
from numpy.random import seed
from numpy import mean
from numpy import var
from math import sqrt
 

def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

effect_sizes = []
for i in range(3): 
    data1 = data[:, i][df_gen.Group == 1][df_gen.Amyloid[df_gen.Group == 1] == 'Positive']
    data2 = data[:, i][df_gen.Group == 1][df_gen.Amyloid[df_gen.Group == 1] == 'Negative']
    # data1 = data[:, i][df_gen.Group == 1][df_gen.APOEe4[df_gen.Group == 1] == 0]
    # data2 = data[:, i][df_gen.Group == 1][df_gen.APOEe4[df_gen.Group == 1] == 4]
    d = cohend(data1, data2)
    effect_sizes.append(d)
    print('Cohens d: %.3f' % d)
    # alpha = 0.05
    # power = 0.8
    # analysis = TTestIndPower()
    # result = analysis.solve_power(d, power=power, nobs1=None, ratio=1.0, alpha=alpha)
    # print('Min sample Size: %.3f' % result)

# calculate power curves for varying sample and effect size
from numpy import array
from matplotlib import pyplot
from statsmodels.stats.power import TTestIndPower
# parameters for power analysis
effect_sizes = array(effect_sizes)
sample_sizes = array(range(5, data.shape[0]))
# calculate power curves from multiple power analyses
analysis = TTestIndPower()
analysis.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
pyplot.show()


###############################
# COMPARISON ATROPHY PATTERNS #
###############################

pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

x_loadings, y_loadings, lv_sulci, lv_cogn = [], [], [], []
boostx, boosty = [], []
for database, confound, train_hc_only in zip(['pisa', 'aibl', 'pisa'],
                                             [False, False, True],
                                             [True, True, False]):
    df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                               cogn_startswith=cogn_startswith)
    if database == 'aibl':
        df_gen['Group'] = [1 if c == 'HC' else c for c in df_gen.classification]
        df_gen['Group'] = [2 if c in ['MCI', 'MCIX'] else c for c in df_gen.Group]
        df_gen['Group'] = [3 if c in ['AD', 'DNOS', 'FTD', 'PD', 'VD'] else c for c in df_gen.Group]
        df_sulci = df_sulci.loc[[idx for idx in df_sulci.index if idx.endswith('_bl')]]
        df_cogn = df_cogn.loc[[idx for idx in df_cogn.index if idx.endswith('_bl')]]
        df_gen = df_gen.loc[[idx for idx in df_gen.index if idx.endswith('_bl')]]
    if remove_missing:
        df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
    if downsampling:
        df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
    df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
        df_sulci, df_cogn, df_gen, confound, train_hc_only)

    Xsulci_r, Ycogn_r = pipeline.fit_transform(Xsulci, Ycogn)
    x_loadings.append([pearsonr(Xsulci_r[:, 0], Xsulci[col].fillna(Xsulci[col].mean()))[0] for col in df_sulci.columns])
    y_loadings.append([pearsonr(Ycogn_r[:, 0], Ycogn[col].fillna(Ycogn[col].mean()))[0] for col in df_cogn.columns])
    Xsulci_r, Ycogn_r = pipeline.transform(df_sulci, df_cogn)
    lv_sulci.append(Xsulci_r)
    lv_cogn.append(Ycogn_r)
    xw, yw, xl, yl = pls.bootstrapping(Xsulci, Ycogn, pipeline)
    boostx.append(xl)
    boosty.append(yl)


np.corrcoef(x_loadings)
# aibl ressemble plus a pisa apres regression out que sans regression out
plt.plot(x_loadings[0], x_loadings[1], 'o')
plt.plot(x_loadings[1], x_loadings[2], 'o')
plt.plot(x_loadings[0], x_loadings[2], 'o')

# snapshot
from uon.snapshots import view_sulcus_scores
scores = np.array(x_loadings)
sc = scores[0] - scores[2]
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
            # minVal=0, maxVal=0.2,
            # background=[0,0,0,1],
            snapshot=f'/tmp/{side}{reg}_bootstrap.png')

full_labels = Ycogn.columns
labels = [s[s.rfind('.')+1:] if '.' in s else s for s in full_labels] 
scores = y_loadings
means = np.array(scores[0]) - np.array(scores[2])
full_labels = Ycogn.columns
labels = [s[s.rfind('.')+1:] if '.' in s else s for s in full_labels] 
labels = [l for m, l in sorted(zip(means, labels))]
full_labels = [l for m, l in sorted(zip(means, full_labels))]
probas = sorted(means)
colors = []
for l in full_labels:
    if l.startswith('M.'):
        colors.append('lightcoral')
    elif l.startswith('L.'):
        colors.append('lightgreen')
    elif l.startswith('E.'):
        colors.append('lightblue')
    elif l.startswith('S.'):
        colors.append('plum')
    else:
        colors.append('moccasin')
plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(5, 20))
plt.barh(labels, probas, 0.5, color=colors)

# plot lv
plot_values(lv_sulci[0][:, 0], lv_sulci[2][:, 0], df_gen.Group,
            col={1: 'r', 2: 'g', 3: 'b'},
            dict_label={1: 'HC', 2: 'MCI', 3: 'AD'},
            invert_axes=True, xlabel='SW HC only', ylabel='SW regress out')
plot_values(lv_cogn[0][:, 0], lv_cogn[2][:, 0], df_gen.Group,
            col={1: 'r', 2: 'g', 3: 'b'},
            dict_label={1: 'HC', 2: 'MCI', 3: 'AD'},
            invert_axes=True, xlabel='COGN HC only', ylabel='COGN regress out')

# boost
import matplotlib.patches as mpatches
plt.figure(figsize=(20, 12))
labels = []
violin = plt.violinplot(np.array(boosty[0]).T[0].T)
color = violin["bodies"][0].get_facecolor().flatten()
labels.append((mpatches.Patch(color=color), 'PISA healthy ageing'))
violin = plt.violinplot(np.array(boosty[2]).T[0].T)
color = violin["bodies"][0].get_facecolor().flatten()
labels.append((mpatches.Patch(color=color), 'PISA pathologic ageing'))
plt.xticks(range(1, df_cogn.shape[1]+1), df_cogn.columns, rotation='vertical')
plt.legend(*zip(*labels), loc=2)

plt.figure(figsize=(20, 12))
plt.violinplot(np.array(boostx[0]).T[0].T)
plt.violinplot(np.array(boostx[2]).T[0].T)
plt.xticks(range(1, df_sulci.shape[1]+1), df_sulci.columns, rotation='vertical')

for i in range(df_sulci.shape[1]):
    st, pval = stats.ttest_ind(np.array(boostx[0])[:, i, 0],
                    np.array(boostx[2])[:, i, 0])
    if pval < 0.05:
        print(df_sulci.columns[i], pval, st)

for i in range(df_cogn.shape[1]):
    st, pval = stats.ttest_ind(np.array(boosty[0])[:, i, 0],
                    np.array(boosty[2])[:, i, 0])
    if pval < 0.05:
        print(df_cogn.columns[i], pval, st)

###########################
# TEST GENETIC PROPORTION #
###########################
from sklearn import preprocessing

database = 'pisa'

# NORMAL AGEING
train_hc_only = True
confound = False

df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                           cogn_startswith=cogn_startswith)
if remove_missing:
    df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
if downsampling:
    df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
    df_sulci, df_cogn, df_gen, confound, train_hc_only)

plsca = PLSCanonical(n_components=1, scale=False)
Xscaler = preprocessing.StandardScaler()
Yscaler = preprocessing.StandardScaler()
Ximputer = SimpleImputer(strategy="mean")
Yimputer = SimpleImputer(strategy="mean")

Xscaler.fit(df_sulci[df_gen.Group == 1])
X = Xscaler.transform(df_sulci)
Yscaler.fit(df_cogn[df_gen.Group == 1])
Y = Yscaler.transform(df_cogn)
Ximputer.fit(X[df_gen.Group == 1])
X = Ximputer.transform(X)
Yimputer.fit(Y[df_gen.Group == 1])
Y = Yimputer.transform(Y)
plsca.fit(X[df_gen.Group == 1], Y[df_gen.Group == 1])

Xall = Xscaler.transform(df_sulci)
Yall = Yscaler.transform(df_cogn)
Xall = Ximputer.transform(Xall)
Yall = Yimputer.transform(Yall)

Xr_ha, Yr_ha = plsca.transform(Xall, Yall)

# AD AGEING
train_hc_only = False
confound = True

df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                           cogn_startswith=cogn_startswith)
if remove_missing:
    df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
if downsampling:
    df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
    df_sulci, df_cogn, df_gen, confound, train_hc_only)

plsca = PLSCanonical(n_components=1, scale=False)
Xscaler = preprocessing.StandardScaler()
Yscaler = preprocessing.StandardScaler()
Ximputer = SimpleImputer(strategy="mean")
Yimputer = SimpleImputer(strategy="mean")

Xscaler.fit(df_sulci) ## not sure
X = Xscaler.transform(df_sulci)
Yscaler.fit(df_cogn)
Y = Yscaler.transform(df_cogn)
Ximputer.fit(X)
X = Ximputer.transform(X)
Yimputer.fit(Y)
Y = Yimputer.transform(Y)
plsca.fit(X, Y)

# Xall = Xscaler.transform(df_sulci)
# Yall = Yscaler.transform(df_cogn)
# Xall = Ximputer.transform(Xall)
# Yall = Yimputer.transform(Yall)

Xr_ad, Yr_ad = plsca.transform(Xall, Yall)

# plot lv
plot_values(Xr_ha[:, 0], Xr_ad[:, 0], df_gen.Group,
            col={1: 'r', 2: 'g', 3: 'b'},
            dict_label={1: 'HC', 2: 'MCI', 3: 'AD'},
            invert_axes=True, xlabel='SW HC only', ylabel='SW regress out')
plot_values(Yr_ha[:, 0], Yr_ad[:, 0], df_gen.Group,
            col={1: 'r', 2: 'g', 3: 'b'},
            dict_label={1: 'HC', 2: 'MCI', 3: 'AD'},
            invert_axes=True, xlabel='COGN HC only', ylabel='COGN regress out')
plot_values(Xr_ha[:, 0]-Xr_ad[:, 0], df_gen.Age, df_gen.Group,
            col={1: 'r', 2: 'g', 3: 'b'},
            dict_label={1: 'HC', 2: 'MCI', 3: 'AD'},
            invert_axes=True, xlabel='COGN HC only', ylabel='COGN regress out')
plot_values(Yr_ha[:, 0]-Yr_ad[:, 0], df_gen.Age, df_gen.Group,
            col={1: 'r', 2: 'g', 3: 'b'},
            dict_label={1: 'HC', 2: 'MCI', 3: 'AD'},
            invert_axes=True, xlabel='COGN HC only', ylabel='COGN regress out')

# TEST GENETIC
df_gen['APOEe4'] = [4 if a in [24, 34, 44] else a for a in df_gen.APOEv3]
df_gen['APOEe4'] = [a if a == 4 else 0 for a in df_gen.APOEe4]
df_gen['APOEe2'] = [2 if a in [24, 23] else a for a in df_gen.APOEv3]
df_gen['APOEe2'] = [a if (np.isnan(a) or a == 2) else 0 for a in df_gen.APOEe2]
df_gen['Amyloid'] = [np.nan if (pd.isnull(a) or a == 'Uncertain') else a for a in df_gen['Amyloid Status']]
risk_group = [1 if x=='low' else x for x in df_gen['classification of risk']]
risk_group = [2 if x in ['high', 'high '] else x for x in risk_group]
df_gen['risk_group'] = np.array([x if x in [1, 2] else np.nan for x in risk_group])

from pingouin import ancova
df = df_gen[['Age', 'Sex', 'Group', 'Amyloid', 'APOEe4']]
df['SW'] = np.array(Xr_ha[:, 0]/Xr_ad[:, 0])
df['COGN'] = np.array(Yr_ha[:, 0]/Yr_ad[:, 0])
for dv in ['SW', 'COGN']:
    ancova(data=df[df.Group == 1], dv=dv, covar=['Age', 'Sex'], between='Amyloid')
    ancova(data=df[df.Group == 1], dv=dv, covar=['Age', 'Sex'], between='APOEe4')
    ancova(data=df[df.Group == 1], dv=dv, covar=['Age', 'Sex'], between='Sex')
    ancova(data=df, dv=dv, covar=['Age', 'Sex'], between='Group')


data = np.hstack([Xr_ha-Xr_ad, 
                  Yr_ha-Yr_ad])
plot_groups(data, 
            np.array(df_gen['Group']), 
            title=[f'SW', 'COGN'])

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.APOEe4[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN'],
            col={0:'k', 4:'y'}, 
            dict_label={0:'e4 non carrier', 4:'e4 carrier'},
            invert_axes=True)

plot_values(data[df_gen.Group == 1], np.array(df_gen.Age[df_gen.Group == 1]), 
            groups=df_gen.risk_group[df_gen.Group == 1],
            xlabel='LV', ylabel='age', title=['SW', 'COGN', 'SW+COGN'],
            col={1:'k', 2:'y'}, 
            dict_label={1:'low risk', 2:'high risk'},
            invert_axes=True)


# PRS
plot_values(data[df_gen.Group == 1], np.array(df_gen.Education[df_gen.Group == 1]), 
            xlabel='LV', ylabel='Education', title=['SW', 'COGN', 'SW+COGN'],
            invert_axes=True)
plot_values(data[df_gen.Group == 1], np.array(df_gen.PRS_noAPOE[df_gen.Group == 1]), 
            xlabel='LV', ylabel='PRS no APOE', title=['SW', 'COGN', 'SW+COGN'],
            invert_axes=True)
for col in ['Amyloid', 'APOEe4', 'Sex', 'APOEe2', 'risk_group']:
    plot_groups(data[df_gen.Group == 1], 
                np.array(df_gen[col])[df_gen.Group == 1], 
                title=[f'LV SW, {col}', f'LV COGN, {col}', f'LV SW+COGN, {col}'])

####################
# DATA AIBL / PISA #
####################

confound = False
confound_cv = False
train_hc_only = True ###
remove_missing = True
downsampling = True
mean=True

databases = ['pisa', 'aibl']
df_sulci_list, df_gen_list = [], []
for database in databases:
    df_sulci, df_cogn, df_gen = prep.load_data(database, mean=mean,
                                               cogn_startswith=None)
    if database == 'aibl':
        df_gen['Group'] = [1 if c == 'HC' else c for c in df_gen.classification]
        df_gen['Group'] = [2 if c in ['MCI', 'MCIX'] else c for c in df_gen.Group]
        df_gen['Group'] = [3 if c in ['AD', 'DNOS', 'FTD', 'PD', 'VD'] else c for c in df_gen.Group]
        df_sulci = df_sulci.loc[[idx for idx in df_sulci.index if idx.endswith('_bl')]]
        df_cogn = df_cogn.loc[[idx for idx in df_cogn.index if idx.endswith('_bl')]]
        df_gen = df_gen.loc[[idx for idx in df_gen.index if idx.endswith('_bl')]]
    if remove_missing:
        df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen, 0.5)
    if downsampling:
        # df_sulci, df_cogn, df_gen = prep.downsampling(df_sulci, df_cogn, df_gen)
        df_sulci, df_cogn, df_gen = prep.downsampling2(df_sulci, df_cogn, df_gen)
    df_sulci_list.append(df_sulci)
    df_gen_list.append(df_gen)

for i in range(2):
    plt.hist(df_gen_list[i][df_gen_list[i].Group == 1]['Age'], alpha=0.5,
              label=databases[i], bins=np.linspace(45,90,19))
    # plt.hist(df_gen_list[i]['Age'], alpha=0.5,
    #          label=databases[i], bins=np.linspace(45,95,21))
plt.legend()
plt.xlim(45, 95)
plt.ylim(0, 150)
plt.title('Age HC')

for i in range(2):
    for g in range(3):
        plt.hist(df_gen_list[i][df_gen_list[i].Group == g+1]['Age'],
                 alpha=0.5, label=f'{databases[i]}, {g}')
plt.legend()
plt.xlim(45, 90)
plt.ylim(0, 225)
plt.title('Age')


for i in range(2):
    plt.hist(df_sulci_list[i].mean(axis=1)[df_gen_list[i].Group == 1],
        alpha=0.5, label=databases[i], bins=np.linspace(0.5,3.5,13))
    # plt.hist(df_sulci_list[i].mean(axis=1), alpha=0.5, label=databases[i],
    #           bins=np.linspace(0.5,3.5,13))
plt.legend()
plt.xlim(0, 4)
plt.ylim(0, 225)
plt.title('SW HC')



######################
# PERMUTATION FIGURE #
######################
confound = False
confound_cv = False
train_hc_only = True
pipeline = pls.PLSPipeline(PLSCanonical(n_components=1),
                           Ximputer=SimpleImputer(strategy="mean"),
                           Yimputer=SimpleImputer(strategy="mean"))

m_scores, m_ref_score = [], []
for cogn_startswith in ['M.', 'E.', 'L.']:
    df_sulci, df_cogn, df_gen = prep.load_data('pisa', cogn_startswith=cogn_startswith)
    df_sulci, df_cogn, df_gen = prep.drop_missing_data(df_sulci, df_cogn, df_gen)
    df_sulci, df_cogn, df_gen = prep.downsampling(df_sulci, df_cogn, df_gen)
    df_sulci, df_cogn, df_gen, Xsulci, Ycogn = prep.XY_definition(
        df_sulci, df_cogn, df_gen, confound, train_hc_only)

    pipeline.fit(Xsulci, Ycogn)
    
    n_comp = pipeline.PLS.n_components
    m_ref_score.append(np.diag(np.cov(pipeline.PLS.x_scores_, pipeline.PLS.y_scores_,
                                    rowvar=False)[:n_comp, n_comp:]))
    
    scores = []
    for i in tqdm(range(100)):
        X = Xsulci
        Y = shuffle(Ycogn)
        pipeline.fit(X, Y)
        scores.append(np.diag(np.cov(
            pipeline.PLS.x_scores_, pipeline.PLS.y_scores_,
            rowvar=False)[:n_comp, n_comp:]))
    m_scores.append(np.array(scores)) 

# figure - all modes
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 24})
p = 0.05
n_comp = len(m_scores)
plt.bar(range(1, n_comp+1),
        [np.quantile(m_scores[i][:, 0], 1-p) for i in range(n_comp)], 
        color='silver', alpha=0.4)
plt.plot(range(1, n_comp+1), [m_ref_score[i] for i in range(n_comp)], 'wo')
plt.ylabel("covariance")
plt.xticks(range(1, n_comp+1),
           ['MEMORY \nmode 1', 'EXECUTIVE \nmode 1', 'LANGUAGE \nmode 1'])
plt.violinplot([m_scores[i][:, 0] for i in range(n_comp)])

