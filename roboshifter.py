import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor
from collections import defaultdict
from scipy.stats import pearsonr
from functools import partial
from multiprocessing import Manager, Pool
from utils import plain_batch_generator
from sklearn.metrics import roc_auc_score
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from data_extraction import DataExtractor
from feature_extraction import FeatureExtractor


robustbase = importr('robustbase')
mass = importr('MASS')

pandas2ri.activate()


def prepare_gauss(cov, df, stat_flag, key, cd, data, fit):
    return Roboshifter.prepare_normalize(df, stat_flag, data['switch'],
                                         Roboshifter.normalize_ellipse, cov)


def prepare_check(df, stat_flag, key, cd, data, fit):
    if fit:
        X, y = Roboshifter.prepare_huber(df, stat_flag)
        X = np.abs(X)

        cd['checks'][key] = Roboshifter.fit_huber(X, y)

    check = cd['checks'][key]

    return Roboshifter.prepare_normalize(df, stat_flag, data['switch'],
                                         partial(Roboshifter.check_normalizer, check), False)


def prepare_line_helper(df, stat_flag, key, cd, data, fit):
    if fit and key not in cd['lines']:
        X, y = Roboshifter.prepare_huber(df, stat_flag)

        cd['lines'][key[:2]] = Roboshifter.fit_huber(X, y)


def prepare_line_gauss(df, stat_flag, key, cd, data, fit):
    prepare_line_helper(df, stat_flag, key, cd, data, fit)

    x, y = df.columns
    params = cd['lines'][key[:2]]

    df[y] -= params['a'] * df[x] + params['b']

    return prepare_gauss(True, df, stat_flag, key, cd, data, fit)


def prepare_line(df, stat_flag, key, cd, data, fit):
    prepare_line_helper(df, stat_flag, key, cd, data, fit)

    line = cd['lines'][key[:2]]

    projection = np.dot(df - np.array([0, line['b']]), np.array([1, line['a']]).T) / \
                 np.sqrt(1 + line['a'] ** 2)

    return prepare_1d_distance(pd.DataFrame(projection, index=df.index), stat_flag,
                               key, cd, data, fit)


def prepare_1d_gauss(df, stat_flag, key, cd, data, fit):
    return Roboshifter.prepare_1d(df, stat_flag, data['switch'], np.abs)


def prepare_1d_distance(df, stat_flag, key, cd, data, fit):
    return Roboshifter.prepare_1d(df, stat_flag, data['switch'], lambda x: x)


def fit_gaussian(df, stat_flag, key, cd, data, fit):
    cd['gaussians'][key] = Roboshifter.mcd(df.loc[stat_flag == 0])


def predict_mahalanobis(df, stat_flag, key, cd, data, fit):
    return Roboshifter.mahalanobis(df, cd['gaussians'][key])


def handle_interactions_helper(pack):
    features, cd, stat_flag, data, fit, handlers, args = pack

    for key, df, interaction in args:
        handler = handlers.get(interaction, lambda *x: x[0])

        new_key = key + (interaction,)
        features[new_key] = handler(df.copy(), stat_flag.copy(), new_key, cd, data, fit)


def predict_euclidean(df, stat_flag, key, cd, data, fit):
    return np.sqrt((df ** 2).sum(axis=1))


class Roboshifter:
    HISTO_FEATURES = {
        'TH1D': [
            ('func', 'mean'),
            ('func', 'var'),
            ('func', 'skew'),
            ('func', 'kurt'),

            ('distance', 'variational_distance'),
            ('distance', 'kolmogorov_smirnov_statistic')
        ],
        'TProfile': [
            ('distance', 'variational'),
            ('distance', 'kolmogorov'),
        ],
        'WeirdTProfile': [
            ('distance', 'variational')
        ],
        'WeirdMuTProfile': [
            ('func', 'mean'),

            ('distance', 'variational'),
            ('distance', 'kolmogorov'),
        ],
        'TEfficiency': [
            ('func', 'mean'),
            ('func', 'var'),
            ('func', 'skew'),
            ('func', 'kurt'),

            ('eff', 'refs - runs'),

            ('eff', 'variational'),
        ]
    }

    SELECTED_LINEAR_FEATURES = [
        'dq_jpsi_yield_err',
        'dq_pi0_mass_err',
        'avHltPhysRate'
    ]

    STATISTICAL_INTEGRAL_FEATURES = {
        'TH1D': [('stats', 'runs'),],
        'TProfile': [('stats', 'runs'),],
        'TEfficiency': [('stats', 'runs'),],
        'WeirdMuTProfile': [('stats', 'runs'),],
        'WeirdTProfile': [('stats', 'runs'),],
    }

    INTERACTION_PREPARE = {
        'mixture_gauss': partial(prepare_gauss, True),
        'gauss': partial(prepare_gauss, False),
        'check': prepare_check,
        'line': prepare_line,
        'line_gauss': prepare_line_gauss,
        '1d_gauss': prepare_1d_gauss,
        '1d_distance': prepare_1d_distance
    }
    INTERACTION_FIT = {
        'gauss': fit_gaussian,
        'line_gauss': fit_gaussian,
        'check': fit_gaussian
    }
    INTERACTION_PREDICT = {
        'gauss': predict_mahalanobis,
        'check': predict_mahalanobis,
        'line_gauss': predict_mahalanobis,
        'mixture_gauss': predict_euclidean
    }

    @staticmethod
    def get_pairs(l):
        return [(a1, a2) for i, a1 in enumerate(l) for a2 in l[i + 1:]]

    def __init__(self, mean_window=5, std_window=10, pollution_rate=0.1,
                 feature_filter_threshold=0.9, feature_threshold_percentile=0.85, njobs=4,
                 roc_auc_threshold=0.6, verbose=True):
        self.mean_window = mean_window
        self.std_window = std_window
        self.pollution_rate = pollution_rate
        self.feature_filter_threshold = feature_filter_threshold
        self.feature_threshold_percentile = feature_threshold_percentile
        self.njobs = njobs
        self.roc_auc_threshold = roc_auc_threshold
        self.verbose = verbose

        self.renew()

        self.de = DataExtractor()

        self.interactions = self.de.get_interactions()

        self.prepare_selected_linear_filter = partial(self.prepare_linear_filter,
                                              Roboshifter.SELECTED_LINEAR_FEATURES)
        self.prepare_all_linear_filter = partial(self.prepare_linear_filter,
            set(FeatureExtractor.NUMS) - set(FeatureExtractor.WEIRD_NUMS))

    def renew(self):
        self.classifier_data = {}

        self.info = {}

        self.fit_X = None
        self.fit_y = None

    @staticmethod
    def reindex(index, series):
        return series.reindex(index).interpolate(method='bfill').interpolate(method='ffill')

    def prepare_features_helper(self, data):
        df = data['df']
        flag = data['flag']

        strong_df = df[flag == 0]

        switch = data['switch'][flag == 0]

        return df, flag, strong_df, switch

    def rolling(self, key, window, switch, series, df):
        # result = Roboshifter.iterate_switches(
        #     series, switch,
        #     lambda curr_switch: curr_switch.rolling(window=window).apply(
        #         lambda x: self.huber(x)[key]))

        result = series.rolling(window=window).apply(lambda x: self.huber(x)[key])

        return Roboshifter.reindex(df.index, result)

    def prepare_features(self, data, tail=None,
                         exclude=None, local_std=True, local_mean=True):
        if self.verbose:
            print 'preparing features'

        if exclude is None:
            exclude = []

        df, flag, strong_df, switch = self.prepare_features_helper(data)

        if tail is not None:
            tail_len = max(self.std_window, self.mean_window) if local_std else self.mean_window

            _, _, tail_strong_df, tail_switch = self.prepare_features_helper(tail)

            sl = slice(-(tail_len - 1), None)

            strong_df = pd.concat([tail_strong_df.iloc[sl], strong_df])
            switch = pd.concat([tail_switch.iloc[sl], switch])

        features = pd.DataFrame()

        for col in df.columns:
            if col not in exclude:
                if local_std:
                    rolling_std = self.rolling('s', self.std_window, switch, strong_df[col], df)
                else:
                    rolling_std = 1

                if local_mean:
                    rolling_mean = self.rolling('mu', self.mean_window, switch, strong_df[col], df)
                else:
                    rolling_mean = 0

                feature = (df[col] - rolling_mean) / rolling_std

                features[col] = feature
            else:
                features[col] = df[col]

        if self.verbose:
            print 'finished preparing features'

        return features

    def normalize_df(self, df):
        df = df.copy()

        for col in df.columns:
            std = df[col].std()

            if std != 0:
                df[col] = (df[col] - df[col].mean()) / std
            else:
                df[col] = 0

        return df

    def get_threshold(self, proba, flag, outlier_quantile):
        df = pd.concat([proba, flag], axis=1, keys=['proba', 'flag']).sort_values('proba')
        index = (df.flag.astype(np.float64).cumsum() / df.flag.sum()) \
            .searchsorted([outlier_quantile], 'right')

        return df.iloc[index[0]].proba

    def predict_filter(self, X, preparer, prefix, y=None):
        if self.verbose:
            print 'predicting {} filter'.format(prefix)

        fit = y is not None

        X = preparer(X, y)
        ndf = pd.DataFrame(index=X.index)

        ndf['proba'] = self.classifier_data['filters'][prefix].predict_proba(X)[:, 1]
        ndf['predict'] = self.classifier_data['filters'][prefix].predict(X)

        if fit:
            self.classifier_data['filter_thresholds'][prefix] = self.get_threshold(ndf.proba, y,
                                                                                   self.pollution_rate)
        ndf['flag'] = (ndf.proba > self.classifier_data['filter_thresholds'][prefix]).astype(
            np.int64)

        for col in ndf:
            name = '{}_{}'.format(prefix, col)
            self.info[fit][name] = ndf[col].rename(name)

        if self.verbose:
            print 'finished predicting {} filter'.format(prefix)

    def fit_filter(self, preparer, model, name):
        if self.verbose:
            print 'fitting {} filter'.format(name)

        X, y = preparer(self.fit_X, self.fit_y), self.fit_y

        self.classifier_data['filters'][name] = model
        self.classifier_data['filters'][name].fit(X, y)

        self.predict_filter(self.fit_X, preparer, name, self.fit_y)

        if self.verbose:
            print 'finished fitting {} filter'.format(name)

    def prepare_stat_filter(self, X, y=None):
        ndf = X[[('linear', 'run_length')]]
        stat_cols = [col for col in X.columns if col[0] == 'my' and
                     any(col[1] == key and col[-2:] in value
                         for key, value in Roboshifter.STATISTICAL_INTEGRAL_FEATURES.items())]
        ndf['runssum'] = X[stat_cols].sum(axis=1)

        return self.normalize_df(ndf)

    def fit_stat_filter(self):
        self.fit_filter(self.prepare_stat_filter,
                        LogisticRegression(class_weight='balanced'),
                        'stat')

    def predict_stat_filter(self, X, y=None):
        self.predict_filter(X, self.prepare_stat_filter, 'stat', y)

    def make_data(self, df, switch, fit):
        if fit:
            tail = None
        else:
            tail = {
                'df': self.fit_X[df.columns],
                'switch': self.fit_X[('linear', 'switch')],
                'flag': self.info[True]['stat_flag']
            }

        data = {
            'df': df,
            'switch': switch,
            'flag': self.info[fit]['stat_flag']
        }

        return data, tail

    def prepare_linear_filter(self, cols, X, y=None):
        fit = y is not None

        df = X[[('linear', f) for f in cols]]

        df = np.abs(self.prepare_features(*self.make_data(df, X[('linear', 'switch')], fit)))

        return df

    def fit_linear_filter(self):
        self.fit_filter(self.prepare_selected_linear_filter,
                        LogisticRegression(class_weight='balanced'),
                        'linear')

        df = self.prepare_all_linear_filter(self.fit_X, self.fit_y)

        self.classifier_data['linear_thresholds'] = \
            self.get_quantile(df, self.feature_threshold_percentile)


    def predict_linear_filter(self, X, y=None):
        self.predict_filter(X, self.prepare_selected_linear_filter, 'linear', y)

        df = self.prepare_all_linear_filter(X)

        linear_prediction = {}

        for key, threshold in self.classifier_data['linear_thresholds'].items():
            linear_prediction[key] = (df[key] > threshold).astype(np.int)

        self.info[False]['linear_prediction'] = linear_prediction

    def fit_feature_filter(self, splitted):
        result = {}

        for ht, features in splitted.items():
            mask = {col: True for col in features}
            for i, col1 in enumerate(features):
                if features[col1].std() < 0.01:
                    mask[col1] = False

                if mask[col1]:
                    for col2 in features.columns[i + 1:]:
                        if mask[col2]:
                            r = Roboshifter.pearsonr(features[col1], features[col2])

                            if r > self.feature_filter_threshold:
                                mask[col2] = False

            result[ht] = [col for col, val in mask.items() if val]

        self.info[True]['feature_filter'] = result

    PACK_SIZE = 100

    def handle_interactions(self, generator, handlers, fit, data=None):
        stat_flag = self.info[fit]['stat_flag'].copy()
        if fit:
            stat_flag[self.fit_y == 1] = 1

        manager = Manager()

        features = manager.dict()
        cd = {key: manager.dict(value) for key, value in self.classifier_data.items()}

        packs = plain_batch_generator(
            generator,
            lambda args: (features, cd, stat_flag, data, fit, handlers, args),
            Roboshifter.PACK_SIZE
        )

        pool = Pool(self.njobs)

        for batch in plain_batch_generator(packs, lambda x: x, self.njobs * 4):
            # for item in batch:
            #     handle_interactions_helper(item)

            pool.map(handle_interactions_helper, batch)

        pool.close()
        pool.join()

        self.classifier_data = {key: dict(value) for key, value in cd.items()}

        return dict(features)

    def prepare_generator(self, data):
        for key, interactions in self.interactions.items():
            hk, cols = key
            if hk in data:
                df = data[hk]
    
                if all(col in df for col in cols):
                    for interaction in interactions:
                        yield key, df[list(cols)], interaction

    def fit_predict_generator(self, data):
        for key, df in data.items():
            yield key[:2], df, key[2]

    def filter_features(self, splitted):
        ff = self.info[True]['feature_filter']
        result = {}

        for ht, df in splitted.items():
            result[ht] = df[ff[ht]]

        return result

    @staticmethod
    def mcd(ellipse):
        result = robustbase.covMcd(ellipse, alpha=0.75)
        result = {key: np.array(result.rx(key)[0]) for key in ['center', 'cov']}

        return result

    @staticmethod
    def huber(gaussian):
        result = mass.huber(gaussian)

        return {key: result.rx(key)[0][0] for key in ['mu', 's']}

    @staticmethod
    def ellipse_callback(ellipse, strong_index, cov):
        params = Roboshifter.mcd(ellipse[strong_index])

        result = ellipse - params['center']

        if cov:
            w, v = np.linalg.eig(params['cov'])

            A = np.dot(np.sqrt(np.diag(1. / w)), v.T)

            result = np.dot(A, result.T).T

        return pd.DataFrame(result, columns=ellipse.columns, index=ellipse.index)

    @staticmethod
    def gaussian_callback(gaussian, strong_index, cov):
        params = Roboshifter.huber(gaussian[strong_index])

        result = gaussian - params['mu']

        if cov:
            result /= params['s']

        return result

    @staticmethod
    def pearsonr(x, y):
        return np.abs(pearsonr(x, y)[0])

    @staticmethod
    def normalize_location(data, stat_flag, threshold, callback, cov):
        if (stat_flag == 0).sum() < threshold:
            return data
        else:
            return callback(data, stat_flag == 0, cov)

    @staticmethod
    def normalize_ellipse(ellipse, stat_flag, cov):
        return Roboshifter.normalize_location(ellipse, stat_flag, 10,
                                              Roboshifter.ellipse_callback, cov)

    @staticmethod
    def normalize_gauss(gauss, stat_flag, cov):
        return Roboshifter.normalize_location(gauss, stat_flag, 5,
                                              Roboshifter.gaussian_callback, cov)

    @staticmethod
    def iterate_switches(df, switches, callback):
        return pd.concat([callback(df.loc[switches == switch]) for switch in switches.unique()],
                         axis=0)

    @staticmethod
    def prepare_normalize(df, stat_flag, switches, normalizer, cov):
        return Roboshifter.iterate_switches(df, switches, lambda curr_switch:
            normalizer(curr_switch, stat_flag.loc[curr_switch.index], cov))

    @staticmethod
    def prepare_huber(df, stat_flag):
        fit_df = df.loc[stat_flag == 0]

        X, y = fit_df.as_matrix().T
        return X.reshape((-1, 1)), y

    @staticmethod
    def check_normalizer(check, df, stat_flag, cov):
        df = df.copy()

        x, y = df.columns

        df[y] -= check['a'] * np.abs(df[x]) + check['b']
        df = Roboshifter.normalize_ellipse(df, stat_flag, cov)

        return df

    @staticmethod
    def prepare_1d(df, stat_flag, switches, callback):
        result = callback(
            Roboshifter.prepare_normalize(df, stat_flag, switches, Roboshifter.normalize_gauss, False))

        return result[result.columns[0]]

    @staticmethod
    def fit_huber(X, y):
        model = HuberRegressor()

        model.fit(X, y)

        return {
            'a': model.coef_[0],
            'b': model.intercept_
        }

    def prepare_ee(self, X, fit):
        if self.verbose:
            print 'preparing EllipticEnvelope'

        cols = [col for col in X.columns if col[0] == 'my' and
                any(col[1] == key and col[-2:] in value
                    for key, value in Roboshifter.HISTO_FEATURES.items())]

        df = X[cols]

        splitted = self.split_features(df)

        if fit:
            self.fit_feature_filter(splitted)

        splitted = self.filter_features(splitted)

        features = self.handle_interactions(self.prepare_generator(splitted),
                                            Roboshifter.INTERACTION_PREPARE, fit,
                                            {'switch': X[('linear', 'switch')]})

        if self.verbose:
            print 'finished preparing EllipticEnvelope'

        return features

    def get_quantile(self, df, quantile):
        n = int(quantile * df.shape[0])

        return {col: df[col].sort_values().iloc[n] for col in df}

    def fit_ee(self):
        if self.verbose:
            print 'fitting EllipticEnvelope'

        features = self.prepare_ee(self.fit_X, True)

        self.handle_interactions(self.fit_predict_generator(features),
                                 Roboshifter.INTERACTION_FIT,
                                 True)

        scores = self.predict_ee_scores(features, True)
        self.info[True]['ee_scores'] = scores

        self.classifier_data['feature_thresholds'] = self.get_quantile(
            pd.DataFrame(scores), self.feature_threshold_percentile)

        if self.verbose:
            print 'finished fitting EllipticEnvelope'

    def fit_histo_weights(self):
        if self.verbose:
            print 'fitting histo weights'

        self.predict_ee(self.fit_X, True)
        ee_prediction = self.info[True]['ee_prediction']

        weights = pd.Series({col: roc_auc_score(self.fit_y, ee_prediction[col])
                             for col in ee_prediction})
        weights = np.maximum(0, weights - self.roc_auc_threshold)
        weights /= weights.sum()

        self.info[True]['histo_weights'] = weights

        if self.verbose:
            print 'finished fitting histo weights'

    def prepare_overall(self, fit):
        return self.normalize_df(pd.concat([self.info[fit]['histo_score_agg'],
                                            self.info[fit]['stat_proba'],
                                            self.info[fit]['linear_proba']], axis=1))

    def fit_overall(self):
        self.predict_histo_score(True)

        X = self.prepare_overall(True)

        model = LogisticRegression(class_weight='balanced')
        model.fit(X, self.fit_y)

        self.classifier_data['final']['model'] = model

    @staticmethod
    def mahalanobis(vs, params):
        centered_vs = vs - params['center']
        vi = params['cov']

        sq_distance = (np.dot(centered_vs, vi) * centered_vs).sum(axis=1)

        assert (sq_distance >= 0).all()

        return np.sqrt(sq_distance)

    def predict_ee_scores(self, features, fit):
        return self.handle_interactions(self.fit_predict_generator(features),
                                        Roboshifter.INTERACTION_PREDICT, fit)

    def predict_ee(self, X, fit=False):
        if self.verbose:
            print 'predicting EllipticEnvelope'

        if fit:
            scores = self.info[True]['ee_scores']
        else:
            col = ('linear', 'switch')
            tailed_X = pd.concat([self.fit_X[self.fit_X[col] == X.iloc[0][col]], X])
            features = self.prepare_ee(tailed_X, False)
            for key, value in features.items():
                features[key] = value.iloc[-X.shape[0]:]

            scores = self.predict_ee_scores(features, False)

        result = defaultdict(lambda: pd.Series(0., X.index))
        counts = defaultdict(int)

        for key, value in scores.items():
            result[key[0]] += (value > self.classifier_data['feature_thresholds'][key]).astype(np.int)
            counts[key[0]] += 1

        for key in result.keys():
            result[key] /= counts[key]

        self.info[fit]['ee_prediction'] = pd.DataFrame(result)

        if self.verbose:
            print 'finished predicting EllipticEnvelope'

    def predict_histo_score(self, fit):
        self.info[fit]['histo_score'] = self.info[fit]['ee_prediction'].multiply(
            self.info[True]['histo_weights'], axis=1)

        self.info[fit]['histo_score_agg'] = self.info[fit]['histo_score'].sum(axis=1)

    def predict_overall(self):
        self.info[False]['overall_score'] = self.classifier_data['final']['model'].predict_proba(
            self.prepare_overall(False))[:, 1]

    def init_fit(self, X, y):
        self.renew()

        self.fit_X = X.copy()
        self.fit_y = y.copy()

        self.info[True] = {}

        for key in ['filters', 'filter_thresholds', 'feature_thresholds',
                    'checks', 'lines', 'gaussians', 'final', 'linear_thresholds']:
            self.classifier_data[key] = {}

    def fit(self, X, y):
        self.init_fit(X, y)

        self.fit_stat_filter()
        self.fit_linear_filter()
        self.fit_ee()
        self.fit_histo_weights()
        self.fit_overall()

    def init_predict(self, X):
        self.info[False] = {}

        return X.copy()

    def predict(self, X):
        X = self.init_predict(X)

        self.predict_stat_filter(X)
        self.predict_linear_filter(X)
        self.predict_ee(X)
        self.predict_histo_score(False)
        self.predict_overall()

    def split_features(self, features):
        histo_mapping = defaultdict(list)
        col_mapping = {}

        for col in features.columns:
            if col[0] == 'my':
                histo_mapping[col[2]].append(col)
                col_mapping[col] = col[3:]

        result = {}

        for histo in histo_mapping.keys():
            result[histo] = features[histo_mapping[histo]].rename(columns=col_mapping)

        return result


if __name__ == '__main__':
    de = DataExtractor()
    df = de.get_train_data()

    rs = Roboshifter()

    X, y = df.drop('flag', axis=1), df.flag

    rs.fit(X.iloc[:300], y.iloc[:300])

    rs.predict(X.iloc[300:500])
