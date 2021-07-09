import pandas as pd
import numpy as np
import os.path as osp
from scipy import stats
import warnings
from uon.confounds import ConfoundRegressor
from collections import Counter

root_path = '/home/leonie/Documents/data'

def preprocessing(df_sulci, df_cogn, df_gen, 
                  regress_out_confounds=False, train_hc_only=True,
                  mean=True, drop_rate=0.5):
    '''
    Data preprocessing for sulcal measures and cognitive scores.
    
    Parameters
    ----------
    df_sulci : pandas.DataFrame
        dataframe containing the sulcal measurements (rows : participants; 
        columns : sulci_side)
        
    df_cogn : pandas.DataFrame
        dataframe containing the cognitive scores (rows : same participants as
        )
    
    Returns
    -------
    
    '''
    # averaging sulcal measurements
    if mean:
        slist = list(set([ss[:ss.find('_')] for ss in df_sulci.columns]))
        dfs = pd.DataFrame(index=df_sulci.index)
        for s in slist:
            if s != 'S.GSM.':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dfs[s] = np.nanmean(
                        df_sulci[[f'{s}_left', f'{s}_right']], axis=1)
            else:
                dfs[s] = df_sulci[f'{s}_left']

    # homogenise number of samples
    df_sulci = df_sulci[df_sulci.index.isin(df_cogn.index)].sort_index()
    df_cogn = df_cogn[df_cogn.index.isin(df_sulci.index)].sort_index()
    df_gen = df_gen[df_gen.index.isin(df_sulci.index)].sort_index()

    # drop features
    print(f'---- Drop when more than {drop_rate*100}% missing values')
    sulci_list = df_sulci.columns
    cogn_list = df_cogn.columns
    df_sulci = df_sulci[df_sulci.columns[df_sulci.isna().sum(axis=0) <
                                         drop_rate*df_sulci.shape[0]]]
    df_cogn = df_cogn[df_cogn.columns[df_cogn.isna().sum(axis=0) <
                                      drop_rate*df_cogn.shape[0]]]
    print(f'Dropped features: {len(sulci_list)-df_sulci.shape[1]} sulci' +
          f'{[s for s in sulci_list if not s in df_sulci.columns]} \n' +
          f' and {len(cogn_list)-df_cogn.shape[1]} cognitives variables'+
          f'{[s for s in cogn_list if not s in df_cogn.columns]} \n')

    # drop subjects
    sulci_shape = df_sulci.shape
    cogn_shape = df_cogn.shape
    df_gen = df_gen[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    df_sulci = df_sulci[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    df_cogn = df_cogn[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    print(f'Dropped subjects: {sulci_shape[0]-df_sulci.shape[0]}')
    print()
    
    # downsampling
    df_sulci = df_sulci[~df_gen.Group.isna()]
    df_cogn = df_cogn[~df_gen.Group.isna()]
    df_gen = df_gen[~df_gen.Group.isna()]
    df_sulci = df_sulci[~df_gen.Age.isna()]
    df_cogn = df_cogn[~df_gen.Age.isna()]
    df_gen = df_gen[~df_gen.Age.isna()]
    if 'site' in df_gen.columns:
        for g, m in Counter(df_gen.site).items():
            if m < 3:
                df_sulci = df_sulci[df_gen.site != g]
                df_cogn = df_cogn[df_gen.site != g]
                df_gen = df_gen[df_gen.site != g]
    print('---- Downsampling')
    test_age(df_gen)
    test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    print()
    print('Removing outliers...')
    while True:
        mean_age = df_gen.Age.mean()
        std_age = df_gen.Age.std()
        min_age = df_gen.Age.min()
        max_age = df_gen.Age.max()
        if min_age < mean_age-2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmin()]], inplace=True)
        elif max_age > mean_age+2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmax()]], inplace=True)
        else:
            break
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')

    pval_age = test_age(df_gen)
    pval_sex = test_sex(df_gen)
    
    df_gen = match_age(df_gen)
    df_gen = match_sex(df_gen)
        
    # homogenise number of samples
    df_sulci = df_sulci[df_sulci.index.isin(df_gen.index)]
    df_cogn = df_cogn[df_cogn.index.isin(df_gen.index)]
    print()
    print('Final...')
    test_age(df_gen)
    test_sex(df_gen)

    # XY definition
    if 'site' in df_gen.columns:
        from neuroCombat import neuroCombat
        covars = df_gen[['site', 'Sex', 'Age', 'Group']]
        data = df_sulci.fillna(df_sulci.mean()).transpose()
        categorical_cols = ['Sex', 'Group']
        continuous_cols = ['Age']
        batch_col = 'site'
        data = neuroCombat(dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols)['data']
        dfs = pd.DataFrame(data.transpose(), index=df_sulci.index, columns=df_sulci.columns)
        dfs[df_sulci.isna()] = np.nan
        df_sulci = dfs
    if regress_out_confounds:
        df_cov = df_gen[['Age', 'Sex']]
        pp_sulci = df_sulci.fillna(df_sulci[df_gen.Group == 1].mean())
        pp_cogn = df_cogn.fillna(df_cogn[df_gen.Group == 1].mean())
        cr = ConfoundRegressor()
        cr.fit(pp_sulci[df_gen.Group == 1], df_cov[df_gen.Group == 1])
        pp_sulci = cr.transform(pp_sulci, df_cov)
        cr.fit(pp_cogn[df_gen.Group == 1], df_cov[df_gen.Group == 1])
        pp_cogn = cr.transform(pp_cogn, df_cov)
        pp_sulci[df_sulci.isna()] = np.nan
        pp_cogn[df_cogn.isna()] = np.nan
        df_sulci = pp_sulci
        df_cogn = pp_cogn

    if train_hc_only:
        Xsulci = df_sulci[df_gen.Group == 1] 
        Ycogn = df_cogn[df_gen.Group == 1]
    else:
        Xsulci = df_sulci 
        Ycogn = df_cogn
    
    return df_sulci, df_cogn, df_gen, Xsulci, Ycogn

def match_age(df_gen):
    print()
    print('Match age...')
    pval_age = test_age(df_gen)
    while pval_age < 0.05:
        rm_hc = df_gen.Age[df_gen.Group == 1].idxmin()
        rm_ad = df_gen.Age[df_gen.Group != 1].idxmax()
        for rm_subj in [rm_hc, rm_ad]:
            df_gen.drop([rm_subj], inplace=True)
        pval_age = test_age(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    return df_gen

def test_age(df_gen):
    st, pval_age = stats.ttest_ind(df_gen['Age'][df_gen.Group == 1],
                                   df_gen['Age'][df_gen.Group != 1])
    print(f"{len(df_gen[df_gen.Group == 1])} HC: {df_gen['Age'][df_gen.Group == 1].mean():.2f} +/- {df_gen['Age'][df_gen.Group == 1].std():.2f}, " +
          f"{len(df_gen[df_gen.Group != 1])} MCI/AD: {df_gen['Age'][df_gen.Group != 1].mean():.2f} +/- {df_gen['Age'][df_gen.Group != 1].std():.2f}, " +
          f"p-value = {pval_age:.4f}")
    return pval_age

def match_sex(df_gen):
    print()
    print('Match sex...')
    pval_sex = test_sex(df_gen)
    while pval_sex < 0.05:
        n_hc = len(df_gen[df_gen.Group == 1])
        n_ad = len(df_gen[df_gen.Group != 1])
        # find main group
        if n_hc > n_ad:
            df = df_gen[df_gen.Group == 1]
            dfo = df_gen[df_gen.Group != 1]
        else:
            df = df_gen[df_gen.Group != 1]
            dfo = df_gen[df_gen.Group == 1]
        # find sex to remove
        if sum(df.Sex)/len(df) > sum(dfo.Sex)/len(dfo): 
            dfs = df[df.Sex == 1]
        else:
            dfs = df[df.Sex == 0]
        # find age to remove
        if df.Age.mean() > dfo.Age.mean():
            rm_subj = dfs.Age.idxmax()
        else:
            rm_subj = dfs.Age.idxmin()
        df_gen.drop([rm_subj], inplace=True)
        pval_sex = test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    return df_gen

def test_sex(df_gen):
    n_hc = len(df_gen[df_gen.Group == 1])
    n_ad = len(df_gen[df_gen.Group != 1])
    nm_hc = df_gen['Sex'][df_gen.Group == 1].sum()
    nm_ad = df_gen['Sex'][df_gen.Group != 1].sum()
    oddr, pval_sex = stats.fisher_exact([[nm_hc, n_hc-nm_hc],
                                         [nm_ad, n_ad-nm_ad]])
    print(f"HC: {nm_hc} M / {n_hc-nm_hc} F, " +
          f"MCI:AD: {nm_ad} M / {n_ad-nm_ad} F, " +
          f"p-value = {pval_sex:.4f}")
    return pval_sex











#################
# OLD FUNCTIONS #
#################


def load_data(dataset,
              measure='opening', mean=True,
              cogn_startswith=None):
    '''
    Parameters
    -----------
    dataset : str {'pisa', 'aibl', 'adni'}
        name of the dataset to load
    '''
    data_path = osp.join(root_path, dataset)

    # morphological data
    sfile = osp.join(
        data_path, f'sMRI/sulci_morphometry_measurements/per_measurement/{measure}.csv')
    if osp.exists(sfile):
        df_ssulci = pd.read_csv(sfile, index_col=0)
        if not mean:
            df_sulci = df_ssulci
        else:
            slist = list(set([ss[:ss.find('_')] for ss in df_ssulci.columns]))
            df_sulci = pd.DataFrame(index=df_ssulci.index)
            for s in slist:
                if s != 'S.GSM.':
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        df_sulci[s] = np.nanmean(
                            df_ssulci[[f'{s}_left', f'{s}_right']], axis=1)
                else:
                    df_sulci[s] = df_ssulci[f'{s}_left']
    else:
        df_sulci = pd.DataFrame()

    # cognitive data
    cfile = osp.join(data_path, 'cognitive_scores/neuropsych.csv')
    if osp.exists(cfile):
        df_cogn = pd.read_csv(cfile, index_col=0)
        if cogn_startswith is not None:
            df_cogn = df_cogn[[col for col in df_cogn.columns
                               if col.startswith(cogn_startswith)]]
    else:
        df_cogn = pd.DataFrame()

    # general data
    gfile = osp.join(data_path, 'cognitive_scores/demographics.csv')
    if osp.exists(gfile):
        df_gen = pd.read_csv(gfile, index_col=0)
    else:
        df_gen = pd.DataFrame()

    # homogenise number of samples
    df_sulci = df_sulci[df_sulci.index.isin(df_cogn.index)]
    df_cogn = df_cogn[df_cogn.index.isin(df_sulci.index)]
    df_gen = df_gen[df_gen.index.isin(df_sulci.index)]

    return df_sulci.sort_index(), df_cogn.sort_index(), df_gen.sort_index()

##################
# MISSING VALUES #
##################


def drop_missing_data(df_sulci, df_cogn, df_gen, drop_rate=0.5):
    '''
    1 nothing dropped / 0 everything
    '''
    
    print(f'---- Drop when more than {drop_rate*100}% missing values')
    # drop features
    sulci_list = df_sulci.columns
    cogn_list = df_cogn.columns
    df_sulci = df_sulci[df_sulci.columns[df_sulci.isna().sum(axis=0) <
                                         drop_rate*df_sulci.shape[0]]]
    df_cogn = df_cogn[df_cogn.columns[df_cogn.isna().sum(axis=0) <
                                      drop_rate*df_cogn.shape[0]]]
    print(f'Dropped features: {len(sulci_list)-df_sulci.shape[1]} sulci' +
          f'{[s for s in sulci_list if not s in df_sulci.columns]} \n' +
          f' and {len(cogn_list)-df_cogn.shape[1]} cognitives variables'+
          f'{[s for s in cogn_list if not s in df_cogn.columns]} \n')

    # drop subjects
    sulci_shape = df_sulci.shape
    cogn_shape = df_cogn.shape
    df_gen = df_gen[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    df_sulci = df_sulci[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    df_cogn = df_cogn[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    print(f'Dropped subjects: {sulci_shape[0]-df_sulci.shape[0]}')
    print()

    return df_sulci, df_cogn, df_gen

################
# DOWNSAMPLING #
################

def downsampling(df_sulci, df_cogn, df_gen):
    df_sulci = df_sulci[~df_gen.Group.isna()]
    df_cogn = df_cogn[~df_gen.Group.isna()]
    df_gen = df_gen[~df_gen.Group.isna()]
    df_sulci = df_sulci[~df_gen.Age.isna()]
    df_cogn = df_cogn[~df_gen.Age.isna()]
    df_gen = df_gen[~df_gen.Age.isna()]
    if 'site' in df_gen.columns:
        for g, m in Counter(df_gen.site).items():
            if m < 3:
                df_sulci = df_sulci[df_gen.site != g]
                df_cogn = df_cogn[df_gen.site != g]
                df_gen = df_gen[df_gen.site != g]
    print('---- Downsampling')
    test_age(df_gen)
    test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    print()
    print('Removing outliers...')
    while True:
        mean_age = df_gen.Age.mean()
        std_age = df_gen.Age.std()
        min_age = df_gen.Age.min()
        max_age = df_gen.Age.max()
        if min_age < mean_age-2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmin()]], inplace=True)
        elif max_age > mean_age+2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmax()]], inplace=True)
        else:
            break
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')

    pval_age = test_age(df_gen)
    pval_sex = test_sex(df_gen)
    
    df_gen = match_age(df_gen)
    df_gen = match_sex(df_gen)
        
    # homogenise number of samples
    df_sulci = df_sulci[df_sulci.index.isin(df_gen.index)]
    df_cogn = df_cogn[df_cogn.index.isin(df_gen.index)]
    print()
    print('Final...')
    test_age(df_gen)
    test_sex(df_gen)
    return df_sulci, df_cogn, df_gen

def match_age(df_gen):
    print()
    print('Match age...')
    pval_age = test_age(df_gen)
    while pval_age < 0.05:
        rm_hc = df_gen.Age[df_gen.Group == 1].idxmin()
        rm_ad = df_gen.Age[df_gen.Group != 1].idxmax()
        for rm_subj in [rm_hc, rm_ad]:
            df_gen.drop([rm_subj], inplace=True)
        pval_age = test_age(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    return df_gen

def match_sex(df_gen):
    print()
    print('Match sex...')
    pval_sex = test_sex(df_gen)
    while pval_sex < 0.05:
        n_hc = len(df_gen[df_gen.Group == 1])
        n_ad = len(df_gen[df_gen.Group != 1])
        nm_hc = df_gen['Sex'][df_gen.Group == 1].sum()
        nm_ad = df_gen['Sex'][df_gen.Group != 1].sum()
        # find main group
        if n_hc > n_ad:
            df = df_gen[df_gen.Group == 1]
            dfo = df_gen[df_gen.Group != 1]
        else:
            df = df_gen[df_gen.Group != 1]
            dfo = df_gen[df_gen.Group == 1]
        # find sex to remove
        if sum(df.Sex)/len(df) > sum(dfo.Sex)/len(dfo): 
            dfs = df[df.Sex == 1]
        else:
            dfs = df[df.Sex == 0]
        # find age to remove
        if df.Age.mean() > dfo.Age.mean():
            rm_subj = dfs.Age.idxmax()
        else:
            rm_subj = dfs.Age.idxmin()
        df_gen.drop([rm_subj], inplace=True)
        pval_sex = test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    return df_gen

   

def downsampling2(df_sulci, df_cogn, df_gen):
    print('---- Downsampling')
    test_age(df_gen)
    test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    print()
    print('Removing outliers...')
    while True:
        mean_age = df_gen.Age.mean()
        std_age = df_gen.Age.std()
        min_age = df_gen.Age.min()
        max_age = df_gen.Age.max()
        if min_age < mean_age-2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmin()]], inplace=True)
        elif max_age > mean_age+2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmax()]], inplace=True)
        else:
            break
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    
    print()
    print('Match age...')
    pval_age = test_age(df_gen)
    while pval_age < 0.05:
        rm_hc = df_gen.Age[df_gen.Group == 1].idxmin()
        rm_ad = df_gen.Age[df_gen.Group != 1].idxmax()
        for rm_subj in [rm_hc, rm_ad]:
            df_gen.drop([rm_subj], inplace=True)
        pval_age = test_age(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    # pval_sex
    print()
    print('Match sex...')
    pval_sex = test_sex(df_gen)
    while pval_sex < 0.05:
        df = df_gen[df_gen.Sex == Counter(df_gen.Sex).most_common(1)[0][0]]
        df = df[df['Amyloid Status'] != 'Positive']
        df = df[df.Group == 1] ## pas pour AIBL?
        rm_hc = df.Age.idxmin()
        df_gen.drop([rm_hc], inplace=True)
        pval_sex = test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')

    # homogenise number of samples
    df_sulci = df_sulci[df_sulci.index.isin(df_gen.index)]
    df_cogn = df_cogn[df_cogn.index.isin(df_gen.index)]
    print()
    print('Final...')
    test_age(df_gen)
    test_sex(df_gen)
    return df_sulci, df_cogn, df_gen

def downsampling3(df_sulci, df_cogn, df_gen):
    print('---- Downsampling')
    test_age(df_gen)
    test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    print()
    print('Removing outliers...')
    while True:
        mean_age = df_gen.Age.mean()
        std_age = df_gen.Age.std()
        min_age = df_gen.Age.min()
        max_age = df_gen.Age.max()
        if min_age < mean_age-2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmin()]], inplace=True)
        elif max_age > mean_age+2.5*std_age:
            df_gen.drop(index=[df_gen.index[df_gen.Age.argmax()]], inplace=True)
        else:
            break
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')

    # # pval_age
    # print()
    # print('Match age...')
    # pval_age = test_age(df_gen)
    # while pval_age < 0.05:
    #     rm_hc = df_gen.Age[df_gen.Group == 1].idxmin()
    #     rm_ad = df_gen.Age[df_gen.Group != 1].idxmax()
    #     for rm_subj in [rm_hc, rm_ad]:
    #         df_gen.drop([rm_subj], inplace=True)
    #     pval_age = test_age(df_gen)
    # print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')
    # pval_sex
    print()
    print('Match sex...')
    pval_sex = test_sex(df_gen)
    while pval_sex < 0.05:
        df = df_gen[df_gen.Sex == 1]
        df = df[df.Group != 1] ## pas pour AIBL?
        if df_gen.loc[df_gen.Group == 1, 'Age'].mean() < df_gen.loc[df_gen.Group != 1, 'Age'].mean():
            rm_hc = df.Age.idxmax()
        else:
            rm_hc = df.Age.idxmin()
        df_gen.drop([rm_hc], inplace=True)
        pval_sex = test_sex(df_gen)
    print(f'TOTAL {len(df_gen)} = {len(df_gen[df_gen.Group == 1])} + {len(df_gen[df_gen.Group != 1])}')

    # homogenise number of samples
    df_sulci = df_sulci[df_sulci.index.isin(df_gen.index)]
    df_cogn = df_cogn[df_cogn.index.isin(df_gen.index)]
    print()
    print('Final...')
    test_age(df_gen)
    test_sex(df_gen)
    return df_sulci, df_cogn, df_gen


# # plot age : HC vs. MCI
# plt.hist(df_gen[df_gen.Group == 1]['Age'], label='HC', alpha=0.8)
# plt.hist(df_gen[df_gen.Group != 1]['Age'], label='MCI/AD', alpha=0.8)
# plt.title('Age')
# plt.legend()
# plt.show()

# # plot sex
# labels = ['HC', 'MCI/AD']
# x = np.arange(len(labels))
# width = 0.35 
# plt.bar(x - width/2,
#         [sum(df_gen[df_gen.Group == 1]['Sex'] == 1),
#           sum(df_gen[df_gen.Group != 1]['Sex'] == 1)], width, label='Men', color='lightblue')
# plt.bar(x + width/2,
#         [sum(df_gen[df_gen.Group == 1]['Sex'] == 0),
#           sum(df_gen[df_gen.Group != 1]['Sex'] == 0)], width, label='Women', color='lightcoral')
# plt.title('Sex')
# plt.legend()
# plt.show()

# # plot age : M vs. F
# plt.hist(df_gen[df_gen.Sex == 0]['Age'], label='Women', alpha=0.8, color='lightcoral')
# plt.hist(df_gen[df_gen.Sex == 1]['Age'], label='Men', alpha=0.8, color='lightblue')
# plt.title('Age')
# plt.legend()
# plt.title('All subjects')
# plt.show()

# plt.hist(df_gen[df_gen.Group == 1][df_gen.Sex[df_gen.Group == 1] == 0]['Age'], label='Women', alpha=0.8, color='lightcoral')
# plt.hist(df_gen[df_gen.Group == 1][df_gen.Sex[df_gen.Group == 1] == 1]['Age'], label='Men', alpha=0.8, color='lightblue')
# plt.title('Age')
# plt.legend()
# plt.title('HC')
# plt.show()

# plt.hist(df_gen[df_gen.Group != 1][df_gen.Sex[df_gen.Group != 1] == 0]['Age'], label='Women', alpha=0.8, color='lightcoral')
# plt.hist(df_gen[df_gen.Group != 1][df_gen.Sex[df_gen.Group != 1] == 1]['Age'], label='Men', alpha=0.8, color='lightblue')
# plt.title('Age')
# plt.legend()
# plt.title('MCI/AD')
# plt.show()

##################
# X/Y DEFINITION #
##################


def XY_definition(df_sulci, df_cogn, df_gen,
                  confound, train_hc_only):
    # df_gen['Age2'] = df_gen['Age']**2
    # df_gen['SexAge'] = df_gen['Age']*df_gen['Sex']
    # df_gen['SexAge2'] = df_gen['Age2']*df_gen['Sex']
    # df_cov = df_gen[['Age', 'SexAge', 'Sex']]
    
    if 'site' in df_gen.columns:
        # https://github.com/Jfortin1/neuroCombat/tree/ac82a067412078680973ddf72bd634d51deae735
        from neuroCombat import neuroCombat, neuroCombatFromTraining
        covars = df_gen[['site', 'Sex', 'Age', 'Group']]
        data = df_sulci.fillna(df_sulci.mean()).transpose()
        categorical_cols = ['Sex', 'Group']
        continuous_cols = ['Age']
        batch_col = 'site'
        data = neuroCombat(dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols)['data']
        dfs = pd.DataFrame(data.transpose(), index=df_sulci.index, columns=df_sulci.columns)
        dfs[df_sulci.isna()] = np.nan
        df_sulci = dfs
    if confound:
        df_cov = df_gen[['Age', 'Sex']]
        pp_sulci = df_sulci.fillna(df_sulci[df_gen.Group == 1].mean())
        pp_cogn = df_cogn.fillna(df_cogn[df_gen.Group == 1].mean())
        cr = ConfoundRegressor()
        cr.fit(pp_sulci[df_gen.Group == 1], df_cov[df_gen.Group == 1])
        pp_sulci = cr.transform(pp_sulci, df_cov)
        cr.fit(pp_cogn[df_gen.Group == 1], df_cov[df_gen.Group == 1])
        pp_cogn = cr.transform(pp_cogn, df_cov)
        pp_sulci[df_sulci.isna()] = np.nan
        pp_cogn[df_cogn.isna()] = np.nan
        df_sulci = pp_sulci
        df_cogn = pp_cogn
    # else:
    #     df_cogn['Age'] = df_gen['Age']
    #     df_cogn['Sex'] = df_gen['Sex']

    if train_hc_only:
        Xsulci = df_sulci[df_gen.Group == 1] 
        Ycogn = df_cogn[df_gen.Group == 1]
    else:
        Xsulci = df_sulci 
        Ycogn = df_cogn

    return df_sulci, df_cogn, df_gen, Xsulci, Ycogn
