import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from uon.confounds import ConfoundRegressor

from sklearn.utils import resample
from sklearn.cross_decomposition import PLSCanonical
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, MultiOutputMixin
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


############
# PIPELINE #
############


class PLSPipeline(BaseEstimator, TransformerMixin, MultiOutputMixin):

    def __init__(self, PLS, Ximputer=None, Yimputer=None):
        self.PLS = PLS
        self.Ximputer = Ximputer
        self.Yimputer = Yimputer
        self.Xscaler = preprocessing.StandardScaler()
        self.Yscaler = preprocessing.StandardScaler()
        self.XConfoundRegressor = None
        self.YConfoundRegressor = None

    def fit(self, X, Y, XC=None, YC=None):
        # Imputers
        if self.Ximputer is not None:
            X = self.Ximputer.fit_transform(X)
        if self.Yimputer is not None:
            Y = self.Yimputer.fit_transform(Y)

        # Confounds regressor
        if XC is not None:
            self.XConfoundRegressor = ConfoundRegressor()
            X = self.XConfoundRegressor.fit_transform(X, XC)
        if YC is not None:
            self.YConfoundRegressor = ConfoundRegressor()
            Y = self.YConfoundRegressor.fit_transform(Y, YC)

        # Scaling
        X = self.Xscaler.fit_transform(X)
        Y = self.Yscaler.fit_transform(Y)

        # PLS
        self.PLS.fit(X, Y)

        return self

    def transform(self, X, Y, XC=None, YC=None):
        # Imputer
        if self.Ximputer is not None:
            X = self.Ximputer.transform(X)
        if self.Yimputer is not None:
            Y = self.Yimputer.transform(Y)

        # Confounds regressor
        if self.XConfoundRegressor is not None:
            X = self.XConfoundRegressor.transform(X, XC)
        if self.YConfoundRegressor is not None:
            Y = self.YConfoundRegressor.transform(Y, YC)

        # Scaling
        X = self.Xscaler.transform(X)
        Y = self.Yscaler.transform(Y)

        return self.PLS.transform(X, Y)

    def fit_transform(self, X, Y, XC=None, YC=None):
        self.fit(X, Y, XC, YC)
        return self.transform(X, Y, XC, YC)

    def predict(self, X, XC=None):
        if self.Ximputer is not None:
            X = self.Ximputer.transform(X)
        if self.XConfoundRegressor is not None:
            X = self.XConfoundRegressor.transform(X, XC)
        return self.PLS.predict(X)

####################
# PERMUTATION TEST #
####################

# https://genomicsclass.github.io/book/pages/permutation_tests.html
# https://pubmed.ncbi.nlm.nih.gov/21044043/
def permutation_test(Xsulci, Ycogn,
                     pipeline=None, score_func=np.cov, n_perm=1000):
    '''
    Parameters
    ----------
    score_func (default: np.cov)
        should be np.cov for PLS and np.corrcoef for CCA
    '''
    if pipeline is None:
        pipeline = PLSPipeline(PLSCanonical(n_components=5),
                               Ximputer=SimpleImputer(strategy="mean"),
                               Yimputer=SimpleImputer(strategy="mean"))

    # ref scores
    x_scores, y_scores = pipeline.fit_transform(Xsulci, Ycogn)
    n_comp = pipeline.PLS.n_components
    ref_score = np.diag(score_func(
        x_scores, y_scores, rowvar=False)[:n_comp, n_comp:])

    # permutation scores
    scores = []
    for i in tqdm(range(n_perm)):
        X = Xsulci
        Y = shuffle(Ycogn, random_state=i)
        x_scores, y_scores = pipeline.fit_transform(X, Y)
        scores.append(np.diag(score_func(
            x_scores, y_scores, rowvar=False)[:n_comp, n_comp:]))
    scores = np.array(scores)

    #p < 0.05 <-> p < 0.05/n_comp after bonferroni correction
    p = 0.05 / n_comp
    scores = np.array(scores)
    up_list = []
    print()
    pvals, zcov = [], []
    for mode in range(n_comp):
        sc = ref_score[mode]
        zsc = (sc-np.mean(scores[:, 0]))/np.std(scores[:, 0])
        up = np.quantile(scores[:, 0], 1-p) # 1-p/n_comp)
        up_list.append(up)
        pvals.append((sum(scores[:, 0] >= sc))/(n_perm))
        zcov.append(zsc)
        # if sc > up:
        #     print(f'Mode {mode} is robust: {sc:.2f} > {up:.2f}')
        # else:
        #     print(f'Mode {mode} is not robust: {sc:.2f} < {up:.2f}')
        # print(f'score={sc:.4f}; z-score={zsc:.4f}; p-value={(sum(scores[:, 0] >= sc))/(n_perm):.4f}')
    rstr = '(1rst mode, p-value='
    if pvals[0]==0:
        rstr += '0'
    else:
        rstr += f'{pvals[0]:.2e}'
    if pvals[0] > 0.05:
        rstr += ')'
    else:
        rstr += f', z-covariance={zcov[0]:.2f}; 2nd mode, p-value='
        if pvals[1]==1:
            rstr += '1)'
        else:
            rstr += f'{pvals[1]:.2e})'
    print(rstr)

    # figure - all modes
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 24})
    plt.bar(range(1, n_comp+1), up_list, color='silver', alpha=0.4)
    plt.plot(range(1, n_comp+1), ref_score, 'bo')
    plt.xlabel("modes")
    plt.ylabel("covariance")
    plt.title('Permutation test')
    plt.xticks(range(1, n_comp+1))
    plt.violinplot(scores)

################
# BOOTSTRAPING #
################


def bootstrapping(Xsulci, Ycogn, pipeline=None, n_boot=1000):
    np.random.seed(0)
    if pipeline is None:
        pipeline = PLSPipeline(PLSCanonical(n_components=2),
                               Ximputer=SimpleImputer(strategy="mean"),
                               Yimputer=SimpleImputer(strategy="mean"))
    Xt, Yt = pipeline.fit_transform(Xsulci, Ycogn)
    
    y_weights, x_weights = [], []
    y_loadings, x_loadings = [], []
    for i in tqdm(range(n_boot)):
        Xr, Yr = resample(Xsulci, Ycogn)
        Xrt, Yrt = pipeline.fit_transform(Xr, Yr)
        yw = pipeline.PLS.y_weights_
        xw = pipeline.PLS.x_weights_
        yl = pipeline.PLS.y_loadings_
        xl = pipeline.PLS.x_loadings_
        # for m in range(Xt.shape[1]):
        #     if np.corrcoef(Xrt[:,m], Xt[:,m])[0,1]<0 and np.corrcoef(Yrt[:,m], Yt[:,m])[0,1]<0:
        #         yw[:, m] = -yw[:, m]
        #         xw[:, m] = -xw[:, m]
        #         yl[:, m] = -yl[:, m]
        #         xl[:, m] = -xl[:, m]
        y_weights.append(yw)
        x_weights.append(xw)
        y_loadings.append(yl)
        x_loadings.append(xl)

    for comp in range(pipeline.PLS.n_components):
        print(f'===== COMPONENT {comp} =====')
        for mean_weight, std_weight, var in sorted(zip(
          np.mean(y_weights, axis=0)[:, comp],
          np.std(y_weights, axis=0)[:, comp], Ycogn.columns)):
            if abs(mean_weight) > abs(std_weight):
                print(f'{mean_weight:.3f} +/- {std_weight:.3f} {var}')

    for comp in range(pipeline.PLS.n_components):
        print(f'===== COMPONENT {comp} =====')
        for mean_weight, std_weight, var in sorted(zip(
          np.mean(x_weights, axis=0)[:, comp],
          np.std(x_weights, axis=0)[:, comp], Xsulci.columns)):
            if abs(mean_weight) > abs(std_weight):
                print(f'{mean_weight:.3f} +/- {std_weight:.3f} {var}')

    rstr = ''
    dict_sulci = {'S.s.P.': 'sub-parietal sulcus',
                  'INSULA': 'insula',
                  'F.Coll.': 'collateral fissure',
                  'S.Cu.': 'cuneal sulcus',
                  'OCCIPITAL': 'occipital lobe',
                  'S.T.i.post.': 'posterior inferior temporal sulcus',
                  'F.I.P.': 'intraparietal sulcus',
                  'F.C.L.p.': 'posterior lateral fissure',
                  'F.I.P.Po.C.inf.': 'superior postcentral intraparietal superior sulcus',
                  'F.C.M.ant.': 'calloso marginal anterior fissure',
                  'F.Cal.ant.-Sc.Cal.': 'calcarine fissure',
                  'F.P.O.': 'parieto-occipital fissure',
                  'S.T.s.': 'superior temporal sulcus'}
    i = 0
    comp = 0
    for mean_weight, std_weight, var in sorted(zip(
        np.mean(x_weights, axis=0)[:, comp],
        np.std(x_weights, axis=0)[:, comp], Xsulci.columns), reverse=True):
          if i != 4:
              if var in dict_sulci.keys():
                  rstr += f'the {dict_sulci[var]}, '
              else:
                  rstr += f'{var}, '
              i += 1
          else:
              if var in dict_sulci.keys():
                  rstr = rstr[:-2] + f' and the {dict_sulci[var]}.'
              else:
                  rstr = rstr[:-2] + f' and {var}.'
              break

    print()
    print(rstr)
    
    return x_weights, y_weights, x_loadings, y_loadings

    # SW weights
    # from uon.snapshots import view_sulcus_scores
    # import sigraph
    # scores = [m if up*down > 0 else 0 for m, up, down in zip(
    #     np.mean(x_weights, axis=0)[:, 0],
    #     np.percentile(x_weights, 97.5, axis=0)[:, 0],
    #     np.percentile(x_weights, 2.5, axis=0)[:, 0])]
    # dict_sulcus = {s+'_left': x for s,x in zip(Xsulci.columns, scores)}
    # for s in Xsulci.columns:
    #     dict_sulcus[s+'_left'] = dict_sulcus[s+'_left']
    # dict_reg = {0 : [0.5, -0.5, -0.5, 0.5], 1 : [0.5, 0.5, 0.5, 0.5]}
    # for side in ['left']:
    #     for reg in [0, 1]:
    #         view_sulcus_scores(
    #             dict_sulcus,
    #             side=side,
    #             reg_q=dict_reg[reg], snapshot=f'/tmp/{side}{reg}_boot.png',
    #             background=[0,0,0,1])

##################
# CLASSIFICATION #
##################


def classification(df_sulci, df_cogn, y, pipeline=None):
    if pipeline is None:
        pipeline = PLSPipeline(PLSCanonical(n_components=1),
                               Ximputer=SimpleImputer(strategy="mean"),
                               Yimputer=SimpleImputer(strategy="mean"))
    clf, clf_x, clf_y = SVC(), SVC(), SVC()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores, scores_x, scores_y = [], [], []
    for train, test in skf.split(df_sulci, y):
        Xtrain, Xtest = df_sulci.iloc[train], df_sulci.iloc[test]
        Ytrain, Ytest = df_cogn.iloc[train], df_cogn.iloc[test]
        ytrain, ytest = y[train], y[test]
        # train pipeline on HC train set
        pipeline.fit(Xtrain[ytrain == 0], Ytrain[ytrain == 0])
        # pipeline.fit(Xtrain, Ytrain)
        # train clf on train set
        Xr, Yr = pipeline.transform(Xtrain, Ytrain)
        clf.fit(np.hstack([Xr, Yr]), ytrain)
        clf_x.fit(Xr, ytrain)
        clf_y.fit(Yr, ytrain)
        # test on the test set
        Xrtest, Yrtest = pipeline.transform(Xtest, Ytest)
        y_pred = clf.predict(np.hstack([Xrtest, Yrtest]))
        scores.append(balanced_accuracy_score(ytest, y_pred))
        y_pred = clf_x.predict(Xrtest)
        scores_x.append(balanced_accuracy_score(ytest, y_pred))
        y_pred = clf_y.predict(Yrtest)
        scores_y.append(balanced_accuracy_score(ytest, y_pred))
    scores = np.asarray(scores)
    print(f'Balanced accuracy {scores.mean()*100:.2f} +/- ' +
          f'{scores.std()*100:.2f}')
    scores_x = np.asarray(scores_x)
    print(f'Balanced accuracy with x scores {scores_x.mean()*100:.2f} +/- ' +
          f'{scores_x.std()*100:.2f}')
    scores_y = np.asarray(scores_y)
    print(f'Balanced accuracy with y scores {scores_y.mean()*100:.2f} +/- ' +
          f'{scores_y.std()*100:.2f}')
