import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


robustbase = importr('robustbase')
mass = importr('MASS')

pandas2ri.activate()


def mahalanobis(vs, params):
    centered_vs = vs - params['center']
    vi = params['cov']

    sq_distance = (np.dot(centered_vs, vi) * centered_vs).sum(axis=1)

    assert (sq_distance >= 0).all()

    return np.sqrt(sq_distance)


def reindex(index, series):
    return series.reindex(index).interpolate(method='bfill').interpolate(method='ffill')


def normalize_df(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)


def mcd(ellipse, alpha=0.75):
    result = robustbase.covMcd(ellipse, alpha=alpha)
    result = {key: np.array(result.rx(key)[0]) for key in ['center', 'cov']}

    return result


def huber(gaussian):
    result = mass.huber(gaussian)

    return {key: result.rx(key)[0][0] for key in ['mu', 's']}


def get_linear_features(df, features):
    return df[[('linear', f) for f in features]]


def get_histo_features(df, features):
    cols = [col for col in df.columns if col[0] == 'my' and
            any(col[1] == key and col[-2:] in value
                for key, value in features.items())]

    return df[cols]


def get_histo_feature(df, hks, feature):
    shks = set(hks)
    cols = {col[2]: col for col in df.columns
            if col[0] == 'my' and col[-2:] == feature and col[2] in shks}
    cols = [cols[hk] for hk in hks]

    return df[cols]


class RoboshifterError(Exception):
    pass
