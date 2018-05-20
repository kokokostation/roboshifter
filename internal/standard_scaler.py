import pandas as pd

from roboshifter_utils import huber


class Scaler:
    def __init__(self):
        self.data = None

    def transform(self, X):
        return (X - self.data['mu']) / self.data['s']


class RobustScaler(Scaler):
    def fit(self, X):
        self.data = pd.DataFrame([huber(X[col]) for col in X], index=X.columns)

        return self


class StandardScaler(Scaler):
    def fit(self, X):
        self.data = {
            'mu': X.mean(axis=0, skipna=True),
            's': X.std(axis=0, skipna=True)
        }