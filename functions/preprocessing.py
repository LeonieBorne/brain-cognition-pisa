import pandas as pd
import numpy as np
from scipy import stats
import warnings
from uon.confounds import ConfoundRegressor
from collections import Counter
from neuroCombat import neuroCombat


def preprocessing(df_brain, df_cogn, df_info,  
                  group_col, age_col, sex_col,
                  drop_rate=0.5, regress_out_confounds=False, 
                  train_hc_only=True, mean=True, verbose=True):
    '''
    Data preprocessing for brain and cognitive data.
    
    Parameters
    ----------
    df_brain : pandas.DataFrame
        Brain measurements
        
    df_cogn : pandas.DataFrame
        Cognitive scores
    
    df_info : pandas.DataFrame
        Additional informations which should include age 
        (age_col), sex (sex_col) and clinical status (group_col)
    
    group_col : string
        Name of the column in df_info that contains the clinical status 
        (value should be equal to 1 if the participant is healthy)
    
    age_col : string
        Name of the column in df_info that contains the participants age
        
    sex_col : string
        Name ofthe column in df_info that contains the participants sex
        (1 if Male, 0 if Female)
 
    drop_rate : float, default=0.5
        Maximum percentage of missing values without removing the 
        subject/feature.
        
    regress_out_confounds : bool, default=False
        If True, the confounds (age and sex) are regressed out from the brain
        and cognitive datasets
    
    train_hc_only : bool, default=True
        If True, the training datasets (Xbrain and Ycogn) only contain healthy
        participants
        
    mean : bool, default=True
        If True, the left/right brain measurements are averaged (i.e. the
        measurements named "xxx_left" and "xxx_right" in df_brain)
        
    verbose : bool, default=True
        If True, the different steps are displayed
    
    Returns
    -------
    df_brain : pandas.DataFrame
        Preprocessed brain measurements

    df_cogn : pandas.DataFrame
        Preprocessed cognitives scores
    
    df_info : pandas.DataFrame
        Additional informations
    
    Xbrain : ndarray
        Training preprocessed brain data
    
    Ycogn : ndarray
        Training preprocessed cognitive data
    
    '''
    # averaging left/right measurements
    if mean:
        if verbose:
            print('Averaging left/right measurements...')
        dfs = pd.DataFrame(index=df_brain.index)
        for ss in df_brain.columns:
            s = ss[:ss.find('_')]
            if f'{s}_left' in df_brain.columns and f'{s}_right' in df_brain.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dfs[s] = np.nanmean(
                        df_brain[[f'{s}_left', f'{s}_right']], axis=1)
            else:
                if s == 'S.GSM.':
                    dfs[s] = df_brain[ss]
                else:
                    dfs[ss] = df_brain[ss]
        df_brain = dfs

    # keep subjects present in both df_brain and df_cogn
    if verbose:
        print('Keep subjects present in both datasets...')
    df_brain = df_brain[df_brain.index.isin(df_cogn.index)].sort_index()
    df_cogn = df_cogn[df_cogn.index.isin(df_brain.index)].sort_index()
    df_info = df_info[df_info.index.isin(df_brain.index)].sort_index()
    if verbose:
        print(f'{len(df_brain)} subjects selected.')

    # drop features
    if verbose:
        print(f'Drop features with more than {drop_rate*100}% missing values...')
    sulci_list = df_brain.columns
    cogn_list = df_cogn.columns
    df_brain = df_brain[df_brain.columns[df_brain.isna().sum(axis=0) <
                                         drop_rate*df_brain.shape[0]]]
    df_cogn = df_cogn[df_cogn.columns[df_cogn.isna().sum(axis=0) <
                                      drop_rate*df_cogn.shape[0]]]
    if verbose:
        print(f'{len(sulci_list)-df_brain.shape[1]} brain features dropped: ' +
              f'{[s for s in sulci_list if not s in df_brain.columns]} \n' +
              f'{len(cogn_list)-df_cogn.shape[1]} cognitives features dropped: '+
              f'{[s for s in cogn_list if not s in df_cogn.columns]}')

    # drop subjects
    if verbose:
        print(f'Drop subjects with more than {drop_rate*100}% missing values...')
    nsubj = df_brain.shape[0]
    df_brain = df_brain[df_cogn.isna().sum(axis=1) < drop_rate*df_cogn.shape[1]]
    df_brain = df_brain[df_brain.isna().sum(axis=1) < drop_rate*df_brain.shape[1]]
    df_cogn = df_cogn.loc[df_brain.index]
    df_info = df_info.loc[df_brain.index]
    if verbose:
        print(f'{nsubj-df_brain.shape[0]} dropped subjects')

    # downsampling
    if 'site' in df_info.columns:
        if verbose:
            print('Remove sites with < 3 subjects...')
        for g, m in Counter(df_info.site).items():
            if m < 3:
                df_brain = df_brain[df_info.site != g]
                df_cogn = df_cogn[df_info.site != g]
                df_info = df_info[df_info.site != g]
        if verbose:
            print(f'{len(df_info)} subjects ({sum(df_info.Group==1)}+{sum(df_info.Group!=1)})')
            
    if group_col in df_info.columns and age_col in df_info.columns:
        if verbose:
            print(f'Remove subjects with unknown {group_col} and {age_col}...')
        for col in [age_col, group_col]:
            df_brain = df_brain[~df_info[col].isna()]
            df_cogn = df_cogn.loc[df_brain.index]
            df_info = df_info.loc[df_brain.index]
        if verbose:
            print(f'{len(df_info)} subjects ({sum(df_info.Group==1)}+{sum(df_info.Group!=1)})')
            print('Remove outliers...')
        while True:
            mean_age = df_info.Age.mean()
            std_age = df_info.Age.std()
            min_age = df_info.Age.min()
            max_age = df_info.Age.max()
            if min_age < mean_age-2.5*std_age:
                df_info.drop(index=[df_info.index[df_info.Age.argmin()]], inplace=True)
            elif max_age > mean_age+2.5*std_age:
                df_info.drop(index=[df_info.index[df_info.Age.argmax()]], inplace=True)
            else:
                break
        if verbose:
            print(f'{len(df_info)} subjects ({sum(df_info.Group==1)}+{sum(df_info.Group!=1)})')
            print('Match age...')
        df_info = match_age(df_info, age_col, group_col)
        if verbose:
            print(f'{len(df_info)} subjects ({sum(df_info.Group==1)}+{sum(df_info.Group!=1)})')
            print('Match sex...')
        df_info = match_sex(df_info, sex_col, age_col, group_col)
        if verbose:
            print(f'{len(df_info)} subjects ({sum(df_info.Group==1)}+{sum(df_info.Group!=1)})')
        # homogenise number of samples
        df_brain = df_brain[df_brain.index.isin(df_info.index)]
        df_cogn = df_cogn[df_cogn.index.isin(df_info.index)]

    # remove site effect from brain data
    if 'site' in df_info.columns:
        if verbose:
            print('Apply neuroCombat...')
        covars = df_info[['site', age_col, sex_col, group_col]]
        data = df_brain.fillna(df_brain.mean()).transpose()
        categorical_cols = [sex_col, group_col]
        continuous_cols = [age_col]
        batch_col = 'site'
        data = neuroCombat(dat=data,
            covars=covars,
            batch_col=batch_col,
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols)['data']
        dfs = pd.DataFrame(data.transpose(), index=df_brain.index, columns=df_brain.columns)
        dfs[df_brain.isna()] = np.nan
        df_brain = dfs
    
    # regress out confounds
    if regress_out_confounds:
        if verbose:
            print('Regress out confounds...')
        df_cov = df_info[[age_col, sex_col]]
        pp_sulci = df_brain.fillna(df_brain[df_info[group_col] == 1].mean())
        pp_cogn = df_cogn.fillna(df_cogn[df_info[group_col] == 1].mean())
        cr = ConfoundRegressor()
        cr.fit(pp_sulci[df_info[group_col] == 1], df_cov[df_info[group_col] == 1])
        pp_sulci = cr.transform(pp_sulci, df_cov)
        cr.fit(pp_cogn[df_info[group_col] == 1], df_cov[df_info[group_col] == 1])
        pp_cogn = cr.transform(pp_cogn, df_cov)
        pp_sulci[df_brain.isna()] = np.nan
        pp_cogn[df_cogn.isna()] = np.nan
        df_brain = pp_sulci
        df_cogn = pp_cogn

    # keep only healthy participants
    if train_hc_only:
        if verbose:
            print('Train on healthy participants only...')
        Xbrain = df_brain[df_info[group_col] == 1] 
        Ycogn = df_cogn[df_info[group_col] == 1]
    else:
        Xbrain = df_brain 
        Ycogn = df_cogn
    
    if verbose:
        df_hc = df_info.loc[df_info.Group == 1]
        df_cc = df_info.loc[df_info.Group != 1]
        print()
        print(f'N | {len(df_hc)} HC | {len(df_cc)} CC')
        s = 'Age'
        for dc in [df_hc, df_cc]:
            s += f' | {dc.Age.mean():.0f} [{dc.Age.min():.0f}-{dc.Age.max():.0f}]'
        print(s)
        st, p = stats.ttest_ind(df_hc['Age'], df_cc['Age'])
        print(f'T-test on age means | p{print_pval(p)}, stat={st:.1e}')
        s = 'Males N'
        for dc in [df_hc, df_cc]:
            s += f' | {sum(dc.Sex):.0f}'
        print(s)
        n_hc = len(df_hc)
        n_ad = len(df_cc)
        nm_hc = df_hc['Sex'].sum()
        nm_ad = df_cc['Sex'].sum()
        oddr, p = stats.fisher_exact([[nm_hc, n_hc-nm_hc],
                                      [nm_ad, n_ad-nm_ad]])
        print(f'Fisher exact test (sex) | p{print_pval(p)}, odds-ratio={oddr:.2f}')
        print()
        
    return df_brain, df_cogn, df_info, Xbrain, Ycogn

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

def match_age(df_info, age_col='Age', group_col='Group'):
    pval_age = test_age(df_info, age_col, group_col)
    while pval_age < 0.05:
        df_hc = df_info[df_info[group_col] == 1]
        df_ad = df_info[df_info[group_col] != 1]
        if df_hc[age_col].mean() < df_ad[age_col].mean():
            rm_hc = df_hc[age_col].idxmin()
            rm_ad = df_ad[age_col].idxmax()
        else:
            rm_hc = df_hc[age_col].idxmax()
            rm_ad = df_ad[age_col].idxmin()
        for rm_subj in [rm_hc, rm_ad]:
            df_info.drop([rm_subj], inplace=True)
        pval_age = test_age(df_info, age_col, group_col)
    return df_info

def test_age(df_info, age_col, group_col):
    st, pval_age = stats.ttest_ind(df_info[age_col][df_info[group_col] == 1],
                                   df_info[age_col][df_info[group_col] != 1])
    return pval_age

def match_sex(df_info, sex_col, age_col, group_col):
    pval_sex = test_sex(df_info, sex_col, group_col)
    while pval_sex < 0.05:
        n_hc = len(df_info[df_info[group_col] == 1])
        n_ad = len(df_info[df_info[group_col] != 1])
        # find main group
        if n_hc > n_ad:
            df = df_info[df_info[group_col] == 1]
            dfo = df_info[df_info[group_col] != 1]
        else:
            df = df_info[df_info[group_col] != 1]
            dfo = df_info[df_info[group_col] == 1]
        # find sex to remove
        if sum(df[sex_col])/len(df) > sum(dfo[sex_col])/len(dfo): 
            dfs = df[df[sex_col] == 1]
        else:
            dfs = df[df[sex_col] == 0]
        # find age to remove
        if df[age_col].mean() > dfo[age_col].mean():
            rm_subj = dfs[age_col].idxmax()
        else:
            rm_subj = dfs[age_col].idxmin()
        df_info.drop([rm_subj], inplace=True)
        pval_sex = test_sex(df_info, sex_col, group_col)
    return df_info

def test_sex(df_info, sex_col, group_col):
    n_hc = len(df_info[df_info[group_col] == 1])
    n_ad = len(df_info[df_info[group_col] != 1])
    nm_hc = df_info[sex_col][df_info[group_col] == 1].sum()
    nm_ad = df_info[sex_col][df_info[group_col] != 1].sum()
    oddr, pval_sex = stats.fisher_exact([[nm_hc, n_hc-nm_hc],
                                         [nm_ad, n_ad-nm_ad]])
    return pval_sex
