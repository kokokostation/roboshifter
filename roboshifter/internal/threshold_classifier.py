from utils import nan_gt


class ThresholdClassifier:
    def __init__(self, quantile):
        self.quantile = quantile

        self.reset()

    def reset(self):
        self.thresholds = None

    def get_quantile(self, df):
        n = int(self.quantile * df.shape[0])

        return {col: df[col].sort_values().iloc[n] for col in df}

    def fit(self, X):
        self.thresholds = self.get_quantile(X)

        return self

    def predict(self, X):
        X = X.copy()

        for col in X:
            X[col] = nan_gt(X[col], self.thresholds[col])

        return X