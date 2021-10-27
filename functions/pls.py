from sklearn.base import BaseEstimator, TransformerMixin, MultiOutputMixin
from sklearn import preprocessing

class PLSPipeline(BaseEstimator, TransformerMixin, MultiOutputMixin):

    def __init__(self, PLS, Ximputer=None, Yimputer=None):
        self.PLS = PLS
        self.Ximputer = Ximputer
        self.Yimputer = Yimputer
        self.Xscaler = preprocessing.StandardScaler()
        self.Yscaler = preprocessing.StandardScaler()

    def fit(self, X, Y):
        # Imputers
        if self.Ximputer is not None:
            X = self.Ximputer.fit_transform(X)
        if self.Yimputer is not None:
            Y = self.Yimputer.fit_transform(Y)

        # Scaling
        X = self.Xscaler.fit_transform(X)
        Y = self.Yscaler.fit_transform(Y)

        # PLS
        self.PLS.fit(X, Y)

        return self

    def transform(self, X, Y):
        # Imputer
        if self.Ximputer is not None:
            X = self.Ximputer.transform(X)
        if self.Yimputer is not None:
            Y = self.Yimputer.transform(Y)

        # Scaling
        X = self.Xscaler.transform(X)
        Y = self.Yscaler.transform(Y)

        return self.PLS.transform(X, Y)

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X, Y)

    def predict(self, X):
        if self.Ximputer is not None:
            X = self.Ximputer.transform(X)
        return self.PLS.predict(X)
